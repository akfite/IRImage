classdef IRImage < handle
    %IRIMAGE Class for grayscale image manipulation.
    %   img = IRImage(<MxN array>) creates a reference to an IRImage object
    %   that serves as a container for an image matrix.
    
    properties
        values  = [];
        pads    = [0,0,0,0]; % Top, Bot, Left, Right (in image coordinates... +y points down)
        mirror  = [0,0,0,0];
        padding = NaN; % allow for padding with zeros, nans, whatever
    end
    
    properties (Access = private)
        filepath  = '';
        time      = [];
        az        = [];
        el        = [];
    end

    %% CONSTRUCTOR
    % This constructor can be called with a a 2-D array or a filepath input.
    methods
        function obj = IRImage(inputImage)
            obj = erase(obj);
            
            if nargin < 1, return; end
            
            % first input can be either a filepath or pre-existing values
            if ~isempty(inputImage)
                switch class(inputImage)
                    case {'char', 'string'}
                        % change behavior based on the filepath provided
                        [filepath, filename, ext] = fileparts(inputImage); %#ok<*ASGLU>

                        switch lower(ext)
                            case '.mat'
                                % placeholder
                                error('.mat decoding is not supported at this time.');
                            case '.raw'
                                % might add raw decoding later
                                error('.raw decoding is not supported at this time.');
                            otherwise
                                try
                                    % try to load with imread
                                    img = imread(inputImage);

                                    % make sure it's a grayscale image.  if not, flatten it
                                    if size(img,3) == 3
                                        img = rgb2gray(img);
                                    elseif ~ismatrix(img)
                                        error('Multidimensional data types are not supported.  (NDIMS > 2)')
                                    end

                                    % assign to the object
                                    obj.values = img;
                                catch err
                                    rethrow(err);
                                end
                        end
                        
                        % save the filepath in case we want to reload
                        filepath = inputImage;
                        
                    case 'cell'
                        % when given a cellstr of filepaths, recursively load them into an array of
                        % image objects and return the array
                        
                        % preallocate the array
                        imArray = repmat(obj, numel(inputImage), 1); 
                        
                        % load each cell's contents by calling the class constructor
                        for iFile = 1:numel(inputImage)
                            try
                                imArray(iFile) = IRImage(inputImage{iFile});
                            catch
                                warning('Failed to load %s', inputImage{iFile});
                                imArray(iFile) = IRImage();
                            end
                        end
                            
                        % return the array of objects
                        obj = imArray;
                    otherwise
                        % make sure it's a grayscale image.  if not, flatten it
                        if size(inputImage,3) == 3
                            inputImage = rgb2gray(inputImage);
                        elseif ~ismatrix(inputImage)
                            error('Multidimensional data types are not supported.  (NDIMS > 2)')
                        end
                        
                        % all good.  assign to object
                        obj.values = inputImage;
                                
                end 
            end 
        end
    end
    
    %% SETTORS & ACCESSORS
    methods
        function set.pads(obj, value)
            % function to automatically apply pads to the image when the user
            % changes the 'pads' property value
            pad      = fixPads(value);
            obj      = applyPads(obj, pad);
            obj.pads = pad;
        end
        
        function set.padding(obj, value)
            %PADDING This is a help menu
            % this is more help
            
            if ~isempty(value) && ~isa(value,'cell') && ~isa(value,'struct') && ~isa(value,'sym') && isscalar(value)
                obj.padding = value;
                % now refresh the pads
                applyPads(obj);
            else
                error('Image padding must be defined as a scalar value.');
            end
        end
        
        function set.mirror(obj, value)
        end
        
        function c = rmin(obj)
            % first valid row
            c = 1+abs(obj.pads(1))+abs(obj.mirror(1));
        end
        function c = rmax(obj)
            % last valid row
            c = size(obj.values,1)-abs(obj.pads(2))-abs(obj.mirror(2)); 
        end
        function c = cmin(obj)
            % first valid column
            c = 1+abs(obj.pads(3))+abs(obj.mirror(3));    
        end
        function c = cmax(obj)  
            % last valid column
            c = size(obj.values,2)-abs(obj.pads(4))-abs(obj.mirror(4)); 
        end
    end
    
    %% PUBLIC METHODS
    methods (Access = public)
        
        % ---------------------------------------------------------------------------------------- %
        % RESET BACK TO ORIGINAL STATE
        % ---------------------------------------------------------------------------------------- %
        function obj = reload(obj)
            obj = IRImage(obj.filepath);
        end
        
        % ---------------------------------------------------------------------------------------- %
        % RETURN A NEW OBJECT WITH "clone" OR "copy"
        % ---------------------------------------------------------------------------------------- %
        function newObj = clone(obj), newObj = copy(obj); end   % alternatively use "clone"
        function newObj = copy(obj)
            newObj = IRImage(obj.values);
            newObj.lastpads = obj.pads;
            newObj.lastmirror = obj.mirror;
            newObj.pads = obj.pads;
            newObj.mirror = obj.mirror;
        end
        
        % ---------------------------------------------------------------------------------------- %
        % CUT IMAGE SNIP
        % ---------------------------------------------------------------------------------------- %
        function tile = snip(obj, x, y, snipsize)
            % SNIP Cut out a section of an image.
            %   SNIP = SNIP(OBJ, X, Y, SIZE) cuts out a region of values centered
            %   at coordinate [X,Y] with dimensions defined by SIZE.  Out-of-bounds
            %   values are returned as NaNs.
            %
            %   SIZE can be a 2-element array or a scalar. Scalar input returns a square
            %   box cutout with dimensions of width equal to the scalar.  Inputs are 
            %   required to be odd integer values.
            %
            %   A.Fite, 2017
            
            if nargin < 4 || isempty(snipsize)
                % snip to 10% of the image size
                snipsize = [ceil(max(size(obj.values))*0.1), ceil(max(size(obj.values))*0.1)];
                snipsize = snipsize + (~mod(snipsize,2));
            elseif isscalar(snipsize)
                % make single input a square box of the same width
                snipsize = [snipsize snipsize];
            end
            
            % check for valid input for the size window (odd integers)
            if length(snipsize) ~= 2 || any(isnan(snipsize)) || any(snipsize <= 0)
                error('Invalid SIZE input.  SIZE must be a scalar or 1x2 array of odd integers.');
            end
            
            % force the window size to be odd integers
            snipsize = ceil(snipsize);
            snipsize(mod(snipsize,2) == 0) = snipsize(mod(snipsize,2) == 0) + 1;
            
            % no subpixel shenanigans for this
            x = floor(x);
            y = floor(y);
            
            % initialize the output tile
            tile = nan(snipsize);
            
            % find output tile extents without considering the valid area of the image
            hx = floor(snipsize(1)/2);  
            hy = floor(snipsize(2)/2);
            xi = x-hx; 
            xf = x+hx; 
            yi = y-hy; 
            yf = y+hy;
            
            % now bound the output tile extents to the image boundaries
            Xi = max(xi, 1);
            Xf = min(xf, size(obj.values, 2));
            Yi = max(yi, 1);
            Yf = min(yf, size(obj.values,1));
            
            % place the valid snip area inside the tile, making sure to keep the invalid
            % area consistent with the original image.  i.e. snipping a 3x3 at coordinate (1,1) 
            % should create an invalid region at the top & left sides of the snipped image
            cutout = obj.values(Yi:Yf, Xi:Xf);
            rowrange = 1+(Yi-yi):snipsize(1)-(yf-Yf);
            colrange = 1+(Xi-xi):snipsize(2)-(xf-Xf);
            
            tile(rowrange, colrange) = cutout;
        end
        
        % ---------------------------------------------------------------------------------------- %
        % ROW MEAN REMOVAL
        % ---------------------------------------------------------------------------------------- %
        function obj = rmr(obj)
            obj.values = double(obj.values) - repmat(mean(obj.values,2), [1 size(obj.values,2)]);
        end
        
        % ---------------------------------------------------------------------------------------- %
        % THRESHOLDING
        % ---------------------------------------------------------------------------------------- %
        function obj = thresh(obj, T)
            % THRESH Binarize the image to one or more thresholds.
            %   THRESH(OBJ, T) thresholds the image to the single value or the
            %   array of values in T.  The output image writes pixels starting 
            %   from 0,1,2,...,N.
            %
            %   Ex: binarize the image such that values less than 10 are 0, and values
            %       greater than 10 are 1.
            %
            %           img.thresh(10);
            %
            %   Ex. create a labeled image such that pixels with value...
            %             x  < 5   --> 0
            %       5  <= x < 10  --> 1
            %       10 <= x < 15  --> 2
            %       15 <= x       --> 3
            %
            %           img.thresh([5, 10, 15])
            %
            %   A.Fite, 2017
            outImage = obj.values;
            
            % require that all values be in ascending order
            if ~isvector(T) || any(diff(T) <= 0) || ~all(isreal(T))
                error('Threshold values must be an array of non-repeating, ascending, real numbers.');
            end
            
            % bound the array across all possible values
            try T = [-Inf T Inf]; catch, T = [-Inf; T; Inf]; end
            
            % apply thresholds
            for i = 1:length(T)-1
                outImage(obj.values >= T(i) & obj.values < T(i+1)) = i-1;
            end
            
            obj.values = outImage;
        end
        
        % ---------------------------------------------------------------------------------------- %
        % 1D/2D CORRELATION FILTER
        % ---------------------------------------------------------------------------------------- %
        function obj = filt(obj, filter)
            % FILT Apply a filter to the image.
            %   FILT(OBJ, K) performs the 2D convolution between
            %   the image held by OBJ and the kernel K.
            %
            %   A.Fite, 2017
            
            % apply the filter
            obj.values = conv2(double(obj.values), filter, 'same');
        end
        
        % ---------------------------------------------------------------------------------------- %
        % 1D/2D SUBTRACTIVE CORRELATION FILTER (filter, then subtract from original image)
        % ---------------------------------------------------------------------------------------- %
        function obj = sfilt(obj, filter)
            % SFILT Subtract a filtered image from the original.
            %   SFILT(OBJ, K) performs the 2D convolution between
            %   the image held by OBJ and the kernel K, then
            %   subtracts the result from the original image.
            %
            %   A.Fite, 2017

            % apply the filter, then subtract from the original image
            obj.values = double(obj.values) - conv2(double(obj.values), filter, 'same');
        end
        
        % ---------------------------------------------------------------------------------------- %
        % 2-D FOURIER TRANSFORM
        % ---------------------------------------------------------------------------------------- %
        function out = fft(obj)
            F = fft2(obj.values);
            F = fftshift(F); % center the transform at 0,0
            F = sqrt(real(F).^2 + imag(F).^2); % take the magnitude
            F = log(F+1); % need to reduce the dynamic range and log() is undefined at 0
           
            % only generate a plot if no output is requested
            if nargout > 0
                out = F;
            else
                figure; imagesc(F); colormap gray
            end
        end
        
        % ---------------------------------------------------------------------------------------- %
        % PIXEL-TO-LOS ANGLE CONVERSION
        % ---------------------------------------------------------------------------------------- %
        
        % convert vectors of x & y pixel coordinates into azimuth and elevation vectors
        function [az, el] = pix2los(obj, x, y) %#ok<INUSD,STOUT>
            error('This function is a placeholder for now.');
        end
        
        % convert vectors of los coordinates (az, el) into 
        function [x, y] = los2pix(obj, az, el) %#ok<INUSD,STOUT>
            error('This function is a placeholder for now.');
        end
    end
    
    %% PUBLIC REQUIRING IMAGE TOOLBOX
    methods (Access = public)
        % ---------------------------------------------------------------------------------------- %
        % MEDIAN FILTER
        % ---------------------------------------------------------------------------------------- %
        function obj = med(obj, varargin)
            % MED Apply a median filter to the image.
            %   MED(N,...) applies an NxN median filter.
            %   MED([M N],...) applies an MxN median filter.
            %
            %   The median filter is a non-linear operation that
            %   is typically used to remove noise from an image.
            %   If the image processing toolbox is available, the
            %   user may specifiy any optional inputs that can be
            %   used with the "medfilt2" function. If the toolbox
            %   is not available, it will default to a slower
            %   MATLAB implementation of the median filter with
            %   no options supported.
            %
            %   Ex. 
            %       img = load('clown');
            %       img = IRImage(img.X);
            %       img.med([3 5]);  % apply a 3x5 median filter
            %       img.med(3);      % apply a 3x3 median filter
            %
            %   A.Fite, 2017
            
            if ~isempty(varargin)
                nhood = varargin{1};
                varargin(1) = [];
                
                % allow user to call in [m n] form or just with [m] -> [m m]
                if isscalar(nhood)
                    nhood = [nhood nhood];
                end
            else
                % default when no filter size is specified
                nhood = [3 3];
            end
            
            % check that toolbox exists
            if license('test','image_toolbox')
                obj.values = medfilt2(obj.values, varargin{:});
            else 
                % display a warning only once
                warning('MATLAB:IRImage:MissingToolbox',...
                    'Image Processing Toolbox not found.  Defaulting to inefficient method...');
                warning('off','MATLAB:IRImage:MissingToolbox');
                
                % no image processing toolbox--do it the hard & inefficient way
                hx = floor(nhood(1)/2);
                hy = floor(nhood(2)/2);
                imWidth = size(obj.values,2);
                imHeight = size(obj.values,1);
                
                % initialize output image
                medImage = nan(size(obj.values));
                
                for iRow = 1:imHeight
                    for iCol = 1:imWidth
                        % skip nans (pads)
                        if isnan(obj.values(iRow,iCol)), continue; end
                        
                        % find the median at this point
                        rowrange = max(1, (iRow-hy)):min(imHeight, (iRow+hy));
                        colrange = max(1, (iCol-hx)):min(imWidth, (iCol+hx));
                        pixNHood = obj.values(rowrange, colrange);
                        
                        % write to output image 
                        medImage(iRow,iCol) = median(pixNHood(:),'omitnan');
                    end
                end
                
                % assign to object values
                obj.values = medImage;
            end
        end
    end
     
    %% KERNELS (STATIC METHODS)
    methods (Static, Access = public)
        % ---------------------------------------------------------------------------------------- %
        % BOX (MEAN FILTER) (1D/2D FILTER)
        % ---------------------------------------------------------------------------------------- %
        function kernel = box(M,N)
            % BOX Create a simple box/mean filter kernel.
            %   K = BOX(N) creates an NxN mean filter.
            %   K = BOX([M N]) creates an MxN mean filter.
            %
            %   A.Fite, 2017
            if nargin == 2 && isscalar(N)
                K = [M N];
            elseif isscalar(M)
                K = [M M];
            end
            
            kernel = ones(K);
            kernel = kernel/sum(kernel(:));
        end
        
        % ---------------------------------------------------------------------------------------- %
        % GAUSSIAN BOX FILTER KERNEL (2D FILTER)
        % ---------------------------------------------------------------------------------------- %
        function kernel = gauss2d(N,sigma)
            % GAUSS2D Create a 2-D gaussian filter in pixel space.
            %   K = GAUSS2D(N,SIGMA,RANGE) creates an NxN kernel K which approximates
            %   a multivariate gaussian distribution.  The function is bounded 
            %   on the range [-3,3] in x & y such that with the default SIGMA
            %   value of 1, the gaussian will always precisely fill the kernel.
            %
            %   To apply this kernel to an image inside an IRImage object, use the
            %   primary filtering methods: filt and sfilt. 
            %   e.g.
            %       img = IRImage(...);
            %       img.filt(img.gauss2d(13));
            %
            %   Or, as a static method:
            %       kernel = IRImage.gauss2d(13);
            %
            %   A.Fite, 2017
            
            if isempty(N) || ~isscalar(N) || isnan(N) || N <= 1 || mod(N,2) ~= 1
                error('The kernel width must be an odd integer greater than 1.');
            end
            
            if nargin < 2, sigma = 1; end
            
            % initialize the output kernel 
            kernel = nan(N);
            
            % create a 2D gaussian with mean of zero and standard deviation of 1
            % important to note is that the resolution of the gaussian scales with kernel size
            scalefactor = 25;
            dim = linspace(-3,3,N*scalefactor);
            [x,y] = meshgrid(dim, dim);
            P = 1/(2 * sqrt(2*pi)*sigma) * exp(-(x.^2 + y.^2)./(2*sigma^2));
            
            % now divide the gaussian into a grid the size of the output kernel
            gridStart = 1:scalefactor:(N*scalefactor);
            gridEnd   = scalefactor:scalefactor:(N*scalefactor);
            
            % integrate by summing over each tile in the grid and write the integral to a pixel
            % in the output kernel
            for iRow = 1:N
                for iCol = 1:N
                    activeTile = P(gridStart(iRow):gridEnd(iRow), gridStart(iCol):gridEnd(iCol));
                    kernel(iRow,iCol) = sum(activeTile(:));
                end
            end
            
            % force the kernel to have a unit volume of 1
            kernel = kernel/sum(kernel(:));
        end
        
        % ---------------------------------------------------------------------------------------- %
        % TRIANGLE FILTER
        % ---------------------------------------------------------------------------------------- %
        function kernel = tri(N)
            % TRI Create a 1D triangle filter kernel.
            %   K = TRI(N) creates a 1xN triangle filter by convolving
            %   two box filters.  The output is normalized to have a
            %   unit volume.
            %
            %   A.Fite, 2017
            
            if ~isscalar(N) || N <= 1 || mod(N,2) ~= 1
                error('Invalid input.  N must be an odd integer greater than 1.');
            end
            
            % convolve two box filters to create the triangle filter
            kernel = conv(ones(1,ceil(N/2)), ones(1,ceil(N/2)));
            
            % normalize
            kernel = kernel/sum(kernel(:));
        end
        
    end
    
    %% PRIVATE METHODS
    methods (Access = private)
        % ---------------------------------------------------------------------------------------- %
        % ERASE ALL VALUES
        % ---------------------------------------------------------------------------------------- %
        function obj = erase(obj)
            % public
            obj.values   = [];
            obj.pads     = [0,0,0,0];
            
            % private
            obj.az        = [];
            obj.el        = [];
        end
        
        % ---------------------------------------------------------------------------------------- %
        % APPLY PADS TO THE IMAGE
        % ---------------------------------------------------------------------------------------- %
        function obj = applyPads(obj, newPads)
            if nargin < 2
                % refresh with no args
                newPads = obj.pads;
            end
            
            % calculate the differences to skip sides with no change
            padUpdate = newPads - obj.pads;

            % apply the inner (negative) pads first.  these overwrite part of image
            for side = 1:4             
                if newPads(side) < 0
                    switch side
                        case 1 % TOP (in image coordinates, datum at top-left corner)
                            obj.values((abs(obj.pads(side))+1):(abs(obj.pads(side))-padUpdate(side)), :) = obj.padding;
                        case 2 % BOT
                            obj.values((size(obj.values,1)-abs(obj.pads(side))+padUpdate(side)+1):end, :) = obj.padding;
                        case 3 % LEFT
                            obj.values(:, (abs(obj.pads(side))+1):(abs(obj.pads(side))-padUpdate(side))) = obj.padding;
                        case 4 % RIGHT
                            obj.values(:, (size(obj.values,2)-abs(obj.pads(side))+padUpdate(side)+1):end) = obj.padding;
                    end
                end
            end

            % create a temporary image where we will place the original image on top 
            % (( current size + (positive padding requested) - (previous padding) ))
            tempImageRows = size(obj.values,1)+max(0,newPads(1))+max(0,newPads(2))-max(0,obj.pads(1))-max(0,obj.pads(2));
            tempImageColumns = size(obj.values,2)+max(0,newPads(3))+max(0,newPads(4))-max(0,obj.pads(3))-max(0,obj.pads(4));
            tempImage = repmat(obj.padding, [tempImageRows, tempImageColumns]);

            % for the purposes of placing the original image, negative pads are irrelevant
            tpads = newPads;
            tpads(tpads < 0) = 0;

            % calculate the new indexes into the temp image where the image will live
            tempStartRow = 1+tpads(1);
            tempEndRow   = size(tempImage,1)-tpads(2);
            tempStartCol = 1+tpads(3);
            tempEndCol   = size(tempImage,2)-tpads(4);

            % now need to figure out where the valid portion of the image is in obj.values
            tpadslast = obj.pads;
            tpadslast(tpadslast < 0) = 0;

            % get the indexes to the actual image (minus padding) stored in the object
            objStartRow = 1+tpadslast(1);
            objEndRow   = size(obj.values,1)-tpadslast(2);
            objStartCol = 1+tpadslast(3);
            objEndCol   = size(obj.values,2)-tpadslast(4);

            % place the original image inside the temp image to add the pads
            tempImage(tempStartRow:tempEndRow, tempStartCol:tempEndCol) = obj.values(objStartRow:objEndRow, objStartCol:objEndCol);

            % return the updated image & pads
            obj.values = tempImage;
        end
        
        % ---------------------------------------------------------------------------------------- %
        % APPLY MIRRORING TO THE IMAGE
        % ---------------------------------------------------------------------------------------- %
        function obj = applyMirror(obj, newMirror)
        end
        
        % ---------------------------------------------------------------------------------------- %
        % AUTOMATICALLY REFRESH AXES WHEN DATA CHANGES
        % ---------------------------------------------------------------------------------------- %
        
        function autoUpdateAxes(obj, hImage, prop)
%             % done as 3 separate listeners to preserve backwards compatibility with older MATLAB
%             % versions.  maybe there's a better way to do this.
%             hl_1 = addlistener(obj, 'values', 'PostSet', @(o,e) set(hImage, prop, e.AffectedObject.values));
%             hl_2 = addlistener(obj, 'values', 'PostSet', @(o,e) axis(get(hImage,'parent'),'tight'));
%             hl_3 = addlistener(obj, 'values', 'PostSet', @(o,e) drawnow);
%             
%             % now apply listeners to the image that will remove the image object listeners...
%             addlistener(hImage,'ObjectBeingDestroyed', @(o,e) delete(hl_1));
%             addlistener(hImage,'ObjectBeingDestroyed', @(o,e) delete(hl_2));
%             addlistener(hImage,'ObjectBeingDestroyed', @(o,e) delete(hl_3));
        end
        
    end
    
    %% OVERLOADED METHODS
    methods (Access = public)
        % PLOTTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
        function h = imagesc(obj, varargin) 
            % link the image object to the axis & auto-update when values change
            himg = imagesc(obj.values, varargin{:});
%             autoUpdateAxes(obj, himg, 'cdata');
            if nargout > 0, h = himg; end
        end
        function h = surf(obj, varargin)
            % link the image object to the axis & auto-update when values change
            hsurf = surf(obj.values, varargin{:});
            shading(get(hsurf,'parent'),'interp');  
            axis(get(hsurf,'parent'),'tight');
%             autoUpdateAxes(obj, hsurf, 'zdata');
            if nargout > 0, h = hsurf; end
        end
        function image(obj, varargin),     image(obj.values, varargin{:});      end
        function imshow(obj, varargin),    imshow(obj.values, varargin{:});     end
        function pcolor(obj, varargin),    pcolor(obj.values, varargin{:});     end
        function mesh(obj, varargin),      mesh(obj.values, varargin{:});       end
        function meshc(obj, varargin),     meshc(obj.values, varargin{:});      end
        function meshz(obj, varargin),     meshz(obj.values, varargin{:});      end
        function surfc(obj, varargin),     surfc(obj.values, varargin{:});      end
        function surfl(obj, varargin),     surfl(obj.values, varargin{:});      end
        function waterfall(obj, varargin), waterfall(obj.values, varargin{:});  end
        function hist(obj, varargin),      hist(obj.values(:), varargin{:});    end
        
        % TYPES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
        function obj = double(obj),   obj.values = double(obj.values);        end
        function obj = single(obj),   obj.values = single(obj.values);        end
        function obj = uint8(obj),    obj.values = uint8(obj.values);         end
        function obj = uint16(obj),   obj.values = uint16(obj.values);        end
        function obj = uint32(obj),   obj.values = uint32(obj.values);        end
        function obj = uint64(obj),   obj.values = uint64(obj.values);        end
        function obj = int8(obj),     obj.values = int8(obj.values);          end
        function obj = int16(obj),    obj.values = int16(obj.values);         end
        function obj = int32(obj),    obj.values = int32(obj.values);         end
        function obj = int64(obj),    obj.values = int64(obj.values);         end
        function obj = logical(obj),  obj.values = logical(obj.values);       end
        
        % VALUE RETURN FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
        function v = max(obj),      v = max(obj.values(:));                   end
        function v = min(obj),      v = min(obj.values(:));                   end
        
        % VALUE MODIFICATION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
        
        % MATH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
        % TWO INPUTS
        function obj = plus(a, b),    [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1+arg2;    end
        function obj = minus(a, b),   [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1-arg2;    end
        function obj = times(a, b),   [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1.*arg2;   end
        function obj = mtimes(a, b),  [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1*arg2;    end
        function obj = rdivide(a, b), [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1./arg2;   end
        function obj = ldivide(a, b), [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1.\arg2;   end
        function obj = mrdivide(a, b),[arg1, arg2, obj] = parseargs(a, b); obj.values = arg1/arg2;    end
        function obj = mldivide(a, b),[arg1, arg2, obj] = parseargs(a, b); obj.values = arg1\arg2;    end
        function obj = power(a, b),   [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1.^arg2;   end
        function obj = mpower(a, b),  [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1^arg2;    end
        function obj = lt(a, b),      [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1 < arg2;  end
        function obj = gt(a, b),      [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1 > arg2;  end
        function obj = le(a, b),      [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1 <= arg2; end
        function obj = ge(a, b),      [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1 >= arg2; end
        function obj = ne(a, b),      [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1 ~= arg2; end
        function obj = eq(a, b),      [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1 == arg2; end
        function obj = and(a, b),     [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1 & arg2;  end
        function obj = or(a, b),      [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1 & arg2;  end
        
        % SINGLE INPUT
        function obj = uminus(obj),     obj.values = -obj.values;    end
        function obj = uplus(obj),      obj.values = +obj.values;    end
        function obj = not(obj),        obj.values = ~obj.values;    end
        function obj = ctranspose(obj), obj.values = (obj.values)';  end
        function obj = transpose(obj),  obj.values = (obj.values).'; end
        
        % IMAGE PROCESSING TOOLBOX FUNCTIONS
        function obj = imresize(obj, varargin), obj = resize(obj, varargin); end
        function obj = resize(obj, varargin)
            % check that toolbox exists
            if license('test','image_toolbox')
                obj.values = imresize(obj.values, varargin{:});
            else
                % no image processing toolbox
                stack = dbstack;
                error('The Image Processing Toolbox is required for %s', stack(1).name);
            end
        end
        
        % ---------------------------------------------------------------------------------------- %
        % HELPER FUNCTION TO DEAL WITH DATA TYPES
        % ---------------------------------------------------------------------------------------- %
        function [arg1, arg2, returnObj] = parseargs(a, b)
            % Helper function to pull out the data field from IRImage objects
            
            if isa(b, 'IRImage')
                arg2 = b.values;
                returnObj = b;
            else
                arg2 = b;
            end
            
            if isa(a, 'IRImage')
                arg1 = a.values;
                returnObj = a;
            else
                arg1 = a;
            end
        end
        
        % ---------------------------------------------------------------------------------------- %
        % DISPLAY METHODS
        % ---------------------------------------------------------------------------------------- %
        function disp(obj)
            if isvalid(obj)
                % object still exists
                fprintf('\t<a href="matlab:helpPopup IRImage">IRImage</a> with properties:\n\n');
                fprintf('\t%10s: [%dx%d %s]\n', 'values', size(obj.values,1), size(obj.values,2), class(obj.values));
                if ~isnan(obj.padding) && (mod(obj.padding,1) == 0) % integer type
                    fprintf('\t%10s: [%d %d %d %d] :: %s\n', 'pads', ...
                        obj.pads(1), obj.pads(2), obj.pads(3), obj.pads(4),... 
                        sprintf('<a href="matlab:helpPopup IRImage.set.padding">%d</a>',obj.padding));
                else % floating point type
                    fprintf('\t%10s: [%d %d %d %d] :: %s\n', 'pads', ...
                        obj.pads(1), obj.pads(2), obj.pads(3), obj.pads(4),... 
                        sprintf('<a href="matlab:helpPopup IRImage.set.padding">%.2f</a>',obj.padding));
                end
                fprintf('\t%10s: [%d %d %d %d]\n', 'mirror', obj.mirror(1), obj.mirror(2), obj.mirror(3), obj.mirror(4));
                if ismember(class(obj.values), {'uint8','uint16','uint32','uint64','int8','int16','int32','int64','logical'})
                    fprintf('\t%10s: %d\n', 'max', max(obj.values(:)));
                    fprintf('\t%10s: %d\n', 'min', min(obj.values(:)));
                else
                    fprintf('\t%10s: %3.5f\n', 'max', max(obj.values(:)));
                    fprintf('\t%10s: %3.5f\n', 'min', min(obj.values(:)));
                end
                fprintf('\n');
                fprintf('\tList <a href="matlab: methods(IRImage)">methods</a> for IRImage.\n\n');
            else
                % object has been deleted
                fprintf('\thandle to deleted <a href="matlab:helpPopup IRImage">IRImage</a>\n');
                fprintf('\n');
            end
        end
        
        function methods(obj) %#ok<*MANU>
            fprintf('\n');
            fprintf('  General-purpose methods for class IRImage:');
            fprintf('\n');
            fprintf('\t%51s: %s\n', '<a href="matlab:helpPopup IRImage.filt">filt</a>', 'convolve the image with a kernel');
            fprintf('\t%52s: %s\n', '<a href="matlab:helpPopup IRImage.sfilt">sfilt</a>', 'convolve, then subtract from original image');
            fprintf('\t%51s: %s\n', '<a href="matlab:helpPopup IRImage.snip">snip</a>', 'extract a section of the image');
            fprintf('\t%50s: %s\n', '<a href="matlab:helpPopup IRImage.fft">fft</a>', 'plot the 2D fourier transform');
            fprintf('\t%51s: %s\n', '<a href="matlab:helpPopup IRImage.copy">copy</a>', 'clone the object');
            
            % FILTERS
            fprintf('\n');
            fprintf('  Filters:');
            fprintf('\n');
            fprintf('\t%53s: %s\n', '<a href="matlab:helpPopup IRImage.thresh">thresh</a>', 'threshold to one or more values');
            fprintf('\t%50s: %s\n', '<a href="matlab:helpPopup IRImage.med">med</a>', 'median filter');
            fprintf('\t%50s: %s\n', '<a href="matlab:helpPopup IRImage.rmr">rmr</a>', 'row mean removal');
            
            % KERNELS
            fprintf('\n');
            fprintf('  Kernels (static methods):');
            fprintf('\n');
            fprintf('\t%50s: %s\n', '<a href="matlab:helpPopup IRImage.box">box</a>', 'box blur kernel (mean filter)');
            fprintf('\t%50s: %s\n', '<a href="matlab:helpPopup IRImage.gauss2d">gauss2d</a>', 'gaussian kernel');
            fprintf('\t%50s: %s\n', '<a href="matlab:helpPopup IRImage.tri">tri</a>', 'triangle filter kernel');
            
            fprintf('\n');
        end
    end
    
end

% ------------------------------------------------------------------------------------------------ %
%                                     END OF CLASS DEFINITION
% ------------------------------------------------------------------------------------------------ %

%% LOCAL FUNCTIONS

function pad = fixPads(pad)
% FIXPADS Function to force pads/mirrors into a standard format.
%   PAD = FIXPADS(PAD) performs basic error checking on PAD format
%   and tries to auto-correct invalid input where reasonable.  Basic
%   rules are pads must be real integers.

    % interpret empty arrays as "let's clear this pad out"
    if isempty(pad)
        pad = [0,0,0,0];
    else
        if any(isnan(pad))
            error('Padding values cannot be NaNs!');
        elseif any(~isreal(pad)) || any(~isnumeric(pad))
            error('Padding values must be real integers!');
        elseif ~ismember(length(pad), [1 4])
            error('Padding values can either be 1) empty array, 2) 1x1 scalar, or 3) 1x4 vector.');
        end
    end
    
    % force integers
    pad = floor(pad);
    
    % fix scalar input (1x1 -> 1x4)
    if isscalar(pad)
        pad = repmat(pad, 1, 4);
    end
end

