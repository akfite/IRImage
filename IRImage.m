classdef IRImage < handle
    %IRIMAGE Class for grayscale image manipulation.
    %   IMG = IRImage(<MxN array>) creates a reference to an IRImage object
    %   with three public properties:
    %       1) values
    %       2) mirror   (TOP, BOT, LEFT, RIGHT)
    %       3) pads     (TOP, BOT, LEFT, RIGHT)
    %
    %   While traditionally we would operate on the IMG.values prop, here you 
    %   can interact directly with the IMG object itself.  For example, any math
    %   operation and most plotting functions can be used as follows:
    %       img = IRImage(rand(200,200)); 
    %       img = img * 1.1; 
    %       img.^2;  % image is squared just as if we called "img = img.^2"
    %   Of course, the traditional syntax is still valid:
    %       img.values = img.values * 1.1;
    %   
    %   PADS & MIRRORS
    %   Often when filtering an image we are forced to deal with images whose
    %   valid area shrinks as filters are applied.  Usually the invalid "padded"
    %   region is ignored or 
    
    properties (Access = public, SetObservable)
        values = [];
        mirror = [0,0,0,0];
        pads   = [0,0,0,0]; % Top, Bot, Left, Right (in image coordinates... +y points down)
    end
    
    properties (Access = private)
        azL        = [];
        elL        = [];
        savedData  = [];
        lastpads   = [0,0,0,0]; % the previous "pads" setting
        lastmirror = [0,0,0,0];
    end

    %% CONSTRUCTOR
    % This constructor can be called with a a 2-D array or a filepath input.
    methods
        function obj = IRImage(inputImage)
            obj = erase(obj);
            
            % add listeners
            addlistener(obj, 'pads', 'PostSet', @(src, evnt) obj.padsChanged);   
            addlistener(obj, 'mirror', 'PostSet', @(src, evnt) obj.mirrorChanged);   
            
            if nargin < 1, return; end
            
            % first input can be either a filepath or pre-existing values
            if ~isempty(inputImage)
                switch class(inputImage)
                    case {'char', 'string'}
                        % try to load from file and automatically determine settings based on 
                        % the file extension.  not the most reliable way to load, but convenient
                        filepath = inputImage;
                        if ~exist(filepath,'file')
                            error('Unable to read file ''%s''.  No such file exists.', filepath);
                        end

                        % change behavior based on the filepath provided
                        [~, filename, ext] = fileparts(filepath); %#ok<*ASGLU>

                        switch lower(ext)
                            case {'.tif','.tiff','.png','.jpg','.bmp','.gif','.jpeg','.jp2','.jpx'}
                                % load into a temporary var first
                                img = imread(filepath);

                                % make sure it's a grayscale image.  if not, flatten it
                                if size(img,3) == 3
                                    img = rgb2gray(img);
                                elseif ~ismatrix(img)
                                    error('Multidimensional data types are not supported.  (NDIMS > 2)')
                                end

                                % assign to the object
                                obj.values = img;
                                obj.savedData = obj.values;
                            case '.mat'
                                % placeholder
                                error('.mat decoding is not supported at this time.');
                            case '.raw'
                                % might add raw decoding later
                                error('.raw decoding is not supported at this time.');
                        end
                        
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
                        obj.values = inputImage;
                        obj.savedData = inputImage;
                                
                end 
            end 
        end
    end
    
    %% PUBLIC METHODS
    methods (Access = public)
        
        % ---------------------------------------------------------------------------------------- %
        % RESET BACK TO ORIGINAL STATE
        % ---------------------------------------------------------------------------------------- %
        function obj = reset(obj)
            obj.lastpads   = [0 0 0 0];
            obj.pads       = [0 0 0 0];
            obj.lastmirror = [0 0 0 0];
            obj.mirror     = [0 0 0 0];
            obj.values     = obj.savedData;
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
            if length(snipsize) ~= 2 || ~all(mod(snipsize,2) == 1) || any(isnan(snipsize))
                error('Invalid size window input.  All values must be odd integers.');
            end
            
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
            % THRESH Cut out a section of an image.
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
            %            x  < 5   --> 0
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
            if ~isempty(filter) && isscalar(filter) && ~isnan(filter)
                if mod(filter,2) ~= 0
                    % when given an integer, create a 1-D filter
                    filter = ones(1,filter)/filter;
                else
                    % no funny business
                    error('Filter dimensions must be odd!  %d is not a valid size.',filter)
                end
            end
            
            % apply the filter
            obj.values = filter2(filter, double(obj.values), 'same');
        end
        
        % ---------------------------------------------------------------------------------------- %
        % 1D/2D SUBTRACTIVE CORRELATION FILTER (filter, then subtract from original image)
        % ---------------------------------------------------------------------------------------- %
        function obj = sfilt(obj, filter)
            if ~isempty(filter) && isscalar(filter) && ~isnan(filter)
                % when given an integer, create a 1-D filter
                if mod(filter,2) ~= 0
                    filter = ones(1,filter)/filter;
                else
                    error('Filter dimensions must be odd!  %d is not a valid size.',filter)
                end
            end
            
            % apply the filter
            obj.values = double(obj.values) - filter2(filter, double(obj.values), 'same');
        end
        
        % ---------------------------------------------------------------------------------------- %
        % 2-D FOURIER TRANSFORM
        % ---------------------------------------------------------------------------------------- %
        function out = fft(obj)
            F = fft2(obj.values);
            F = fftshift(F); % center the transform at 0,0
            F = sqrt(real(F).^2 + imag(F).^2); % take the magnitude
            F = log(F+1); % need to reduce the dynamic range and log() is undefined at 0
            figure; imshow(F,[]);
            if nargout > 1, out = F; end
            colormap gray
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
        % RESIZING
        % ---------------------------------------------------------------------------------------- %
        function obj = resize(obj, rows, cols)
            % check that toolbox exists
            if ~license('test','image_toolbox')
                stack = dbstack;
                error('The Image Processing Toolbox is required for %s', stack(1).name);
            end
            
            if nargin < 3 && numel(rows) == 2
                cols = rows(2);
                rows = rows(1);
            end
            
            % if the current image is empty, just initialize to zeros.
            if isempty(obj.values)
                obj.values = zeros(rows, cols);
            else
                % if pads exist, this operation will destroy them.  warn the user
                if any(obj.pads ~= 0)
                    warning('Resizing an image while pads are applied will cause the pads to become a fixed part of the image!');
                    qans = '';
                    while ~strcmpi(qans, 'y') && ~strcmpi(qans, 'n')
                        qans = regexprep(input('Would you still like to resize? (y/n): ','s'), '\w','');
                    end

                    switch qans
                        case 'y'
                            obj.values   = imresize(obj.values, [rows cols]);
                            obj.lastpads = [0 0 0 0];
                            obj.pads     = [0 0 0 0];
                        case 'n'
                            fprintf('Image resize operation aborted with no action taken.\n')
                    end
                end
            end
        end
    end
    
    %% PRIVATE METHODS
    methods (Access = private)
        function obj = erase(obj)
            % public
            obj.values   = [];
            obj.lastpads = [0,0,0,0];
            obj.pads     = [0,0,0,0];
            
            % private
            obj.savedData = [];
            obj.azL       = [];
            obj.elL       = [];
        end
        
        function obj = padsChanged(obj, ~, ~)
            
            % allow the user to input a single value for all the pads, (e.g. pads = 3 --> [3 3 3 3])
            if ~isempty(obj.pads) && isscalar(obj.pads)
                obj.pads = repmat(obj.pads, 1, 4);
            end
            
            % force all integers
            obj.pads = floor(obj.pads);
            
            % have the pads changed value?
            if ~isequal(obj.pads, obj.lastpads)
                % calculate the differences to skip sides with no change
                padUpdate = obj.pads - obj.lastpads;
               
                for side = 1:4
                    if padUpdate(side) == 0
                        continue
                    end
                                        
                    % apply the inner pads.  we'll keep the size the same while replacing parts of
                    % the image with zeros
                    if obj.pads(side) <= 0
                        if (obj.lastpads(side) < 0) && (padUpdate(side) > 0)
                            % x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x 
                            % ABANDONED FOR NOW -- not sure if this is a good idea...
                            %
                            % e.g. the last pad was -5 and we now want a pad of -2.  the region
                            % between -5:-3 has already been removed, so we'll need to use the
                            % saved image state to try and recover that lost information...
                            % x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x 
                            switch side
                                case 1 % TOP (in image coordinates, datum at top-left corner)
                                case 2 % BOT
                                case 3 % LEFT
                                case 4 % RIGHT
                            end
                        elseif obj.pads(side) ~= 0
                            % we are only increasing the padded region, so we don't need to lookup
                            % the saved image state to recover information that was previously
                            % removed by pads.
                            switch side
                                case 1 % TOP (in image coordinates, datum at top-left corner)
                                    obj.values((abs(obj.lastpads(side))+1):(abs(obj.lastpads(side))-padUpdate(side)), :) = NaN;
                                case 2 % BOT
                                    obj.values((size(obj.values,1)-abs(obj.lastpads(side))+padUpdate(side)+1):end, :) = NaN;
                                case 3 % LEFT
                                    obj.values(:, (abs(obj.lastpads(side))+1):(abs(obj.lastpads(side))-padUpdate(side))) = NaN;
                                case 4 % RIGHT
                                    obj.values(:, (size(obj.values,2)-abs(obj.lastpads(side))+padUpdate(side)+1):end) = NaN;
                            end
                        end
                    end
                end
                
                % create a temporary image where we will place the original image on top 
                % (( current size + (positive padding requested) - (previous padding) ))
                tempImageRows = size(obj.values,1)+max(0,obj.pads(1))+max(0,obj.pads(2))-max(0,obj.lastpads(1))-max(0,obj.lastpads(2));
                tempImageColumns = size(obj.values,2)+max(0,obj.pads(3))+max(0,obj.pads(4))-max(0,obj.lastpads(3))-max(0,obj.lastpads(4));
                tempImage = NaN(tempImageRows, tempImageColumns);
                
                % for the purposes of placing the original image, negative pads are irrelevant
                tpads = obj.pads;
                tpads(tpads < 0) = 0;
                
                % calculate the new indexes into the temp image where the image will live
                tempStartRow = 1+tpads(1);
                tempEndRow   = size(tempImage,1)-tpads(2);
                tempStartCol = 1+tpads(3);
                tempEndCol   = size(tempImage,2)-tpads(4);
                
                % now need to figure out where the valid portion of the image is in obj.values
                tpadslast = obj.lastpads;
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
                obj.lastpads = obj.pads;
            end
        end
        
        function obj = mirrorChanged(obj, ~, ~)
            
            if ~isempty(obj.mirror) && isscalar(obj.mirror)
                obj.mirror = repmat(obj.mirror, 1, 4);
            end
            
            if any(obj.mirror < 0)
                obj.mirror = obj.lastmirror;
                error('Mirror values must be positive integers!');
            end
            
            % force all integers
            obj.mirror = floor(obj.mirror);
            
            % have the pads changed value?
            if ~isequal(obj.mirror, obj.lastmirror)
                % calculate the differences to skip sides with no change
                mirrorUpdate = obj.mirror - obj.lastmirror;
                tempImage = zeros(size(obj.values,1)+sum(mirrorUpdate(1:2)), size(obj.values,2)+sum(mirrorUpdate(3:4)));
               
                for side = 1:4
                    if mirrorUpdate(side) == 0
                        continue
                    end
                                        
                    switch side
                        case 1 % TOP (in image coordinates, datum at top-left corner)
                        case 2 % BOT
                        case 3 % LEFT
                        case 4 % RIGHT
                    end
                end
                
                obj.lastmirror = obj.mirror;
            end
        end
        
        % ---------------------------------------------------------------------------------------- %
        % AUTOMATICALLY REFRESH AXES WHEN DATA CHANGES
        % ---------------------------------------------------------------------------------------- %
        
        function autoUpdateAxes(obj, hImage, prop)
            % done as 3 separate listeners to preserve backwards compatibility with older MATLAB
            % versions.  maybe there's a better way to do this.
            hl_1 = addlistener(obj, 'values', 'PostSet', @(o,e) set(hImage, prop, e.AffectedObject.values));
            hl_2 = addlistener(obj, 'values', 'PostSet', @(o,e) axis(get(hImage,'parent'),'tight'));
            hl_3 = addlistener(obj, 'values', 'PostSet', @(o,e) drawnow);
            
            % now apply listeners to the image that will remove the image object listeners...
            addlistener(hImage,'ObjectBeingDestroyed', @(o,e) delete(hl_1));
            addlistener(hImage,'ObjectBeingDestroyed', @(o,e) delete(hl_2));
            addlistener(hImage,'ObjectBeingDestroyed', @(o,e) delete(hl_3));
        end
        
    end
    
    
    %% KERNELS (STATIC METHODS)
    methods (Static, Access = public)
        % ---------------------------------------------------------------------------------------- %
        % BOX (MEAN FILTER) (1D/2D FILTER)
        % ---------------------------------------------------------------------------------------- %
        function kernel = box(M,N)
            % BOX Create a simple box/mean filter kernel.
            %   K = BOX([M N]) creates an MxN mean filter.
            %   K = BOX(N) creates an NxN mean filter.
            %
            %   A.Fite, 2017
            if nargin == 2 && isscalar(N)
                K = [M N];
            elseif isscalar(N)
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
            %   K = GAUSS2D(N,SIGMA) creates an NxN kernel K which approximates
            %   a multivariate Gaussian distribution with a mean of zero and 
            %   a standard deviation of SIGMA.  The function is bounded on the
            %   x/y range [-3,3] regardless of the value of SIGMA.
            %   
            %   When convolved with an image, a 2-D Gaussian filter applies 
            %   a blurring effect similar to a mean filter.  However, in contrast
            %   to a mean filter, the Gaussian is weighted towards the center and
            %   thus is more effective at preserving high-frequency edge content.
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
    end
    
    %% ACCESSORS
    methods (Access = public)
        function v = rmin(obj)
            % first valid row
            v = 1+abs(obj.pads(1))+abs(obj.mirror(1));
        end
        function v = rmax(obj)
            % last valid row
            v = size(obj.values,1)-abs(obj.pads(2))-abs(obj.mirror(2)); 
        end
        function v = cmin(obj)
            % first valid column
            v = 1+abs(obj.pads(3))+abs(obj.mirror(3));    
        end
        function v = cmax(obj)  
            % last valid column
            v = size(obj.values,2)-abs(obj.pads(4))-abs(obj.mirror(4)); 
        end
    end
    
    %% OVERLOADED METHODS
    methods (Access = public)
        % PLOTTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
        function h = imagesc(obj, varargin) 
            % By requesting the plot function on the object rather than the
            % underlying values, we'll add some extra functionality by linking
            % the object to the plot axis and auto-updating the axis as the
            % object changes.
            himg = imagesc(obj.values, varargin{:});
            autoUpdateAxes(obj, himg, 'cdata');
            if nargout > 0, h = himg; end
        end
        function h = surf(obj, varargin)
            hsurf = surf(obj.values, varargin{:});
            shading(get(hsurf,'parent'),'interp');  
            axis(get(hsurf,'parent'),'tight');
            autoUpdateAxes(obj, hsurf, 'zdata');
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
    end
    
end

