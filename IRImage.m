classdef IRImage < hgsetget
    %IRIMAGE IRST+FLIR image manipulation.
    %   IMG = IRImage(<MxN array>) creates a reference to an IRImage object
    %   with three public properties:
    %       1) values
    %       2) mirror   (TOP, BOT, LEFT, RIGHT)
    %       3) pads     (TOP, BOT, LEFT, RIGHT)
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
            addlistener(obj, 'pads', 'PostSet', @IRImage.padsChanged);   
            addlistener(obj, 'mirror', 'PostSet', @IRImage.mirrorChanged);   
            
            if nargin < 1, return; end
            
            % first input can be either a filepath or pre-existing values
            if ~isempty(inputImage)
                switch class(inputImage)
                    case {'char', 'string'}
                        % try to load from file and automatically determine settings based on 
                        % the file extension.  not the most reliable way to load, but convenient
                        obj = loadFromFile(obj, inputImage);
                        
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
        % ROW MEAN REMOVAL
        % ---------------------------------------------------------------------------------------- %
        function obj = rmr(obj)
            obj.values = double(obj.values) - repmat(mean(obj.values,2), [1 size(obj.values,2)]);
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
                    error('1-D filters must have an odd number of elements!  %d is not a valid size.',filter)
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
                    error('1-D filters must have an odd number of elements!  %d is not a valid size.',filter)
                end
            end
            
            % apply the filter
            obj.values = double(obj.values) - filter2(filter, double(obj.values), 'same');
        end
        
        % ---------------------------------------------------------------------------------------- %
        % GAUSSIAN BOX FILTER (2D FILTER)
        % ---------------------------------------------------------------------------------------- %
        function [obj, kernel] = gauss2d(obj, kernelSize)
            % check inputs
            if nargin < 2
                error('Specify the size of the kernel as an odd integer greater than 1.');
            elseif isempty(kernelSize) || ~isscalar(kernelSize) || isnan(kernelSize) || kernelSize <= 1 || mod(kernelSize,2) ~= 1
                error('Invalid kernel size.  The kernel must be an odd integer greater than 1.');
            end
            
            % initialize the kernel in pixel space
            kernel = nan(kernelSize);
            
            % create a 2D gaussian with mean of zero and standard deviation of 1
            % important to note is that the resolution of the gaussian scales with kernel size
            scalefactor = 200;
            dim = linspace(-3,3,kernelSize*scalefactor);
            [x,y] = meshgrid(dim, dim);
            P = 1/(2 * sqrt(2*pi)) * exp(-(x.^2 + y.^2)./(2));
            
            % now divide the gaussian into a grid the size of the output kernel, in pixels
            gridStart = 1:scalefactor:(kernelSize*scalefactor);
            gridEnd   = scalefactor:scalefactor:(kernelSize*scalefactor);
            
            % now integrate by summing over each tile in the grid
            for iRow = 1:kernelSize
                for iCol = 1:kernelSize
                    activeTile = P(gridStart(iRow):gridEnd(iRow), gridStart(iCol):gridEnd(iCol));
                    kernel(iRow,iCol) = sum(activeTile(:));
                end
            end
            
            % force the kernel to have a unit volume of 1
            kernel = kernel/sum(kernel(:));
             
            % now apply the kernel to the values in the object
            obj.values = filter2(kernel, double(obj.values), 'same');
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
    
    %% PUBLIC REQUIRING TOOLBOX
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
        
        function obj = loadFromFile(obj, filepath)
            % if the image doesn't exist, return an error
            if ~exist(filepath,'file')
                error('Unable to read file ''%s''.  No such file exists.', filepath);
            end
            
            % change behavior based on the filepath provided
            [~, filename, ext] = fileparts(filepath); %#ok<*ASGLU>
            
            switch ext
                case {'.tif','.tiff','.png','.jpg','.bmp','.gif','.jpeg','.jp2','.jpx'}
                    obj.values = imread(filepath);
                    obj.savedData = obj.values;
                case '.mat'
                    % placeholder
                case '.raw'
                    % might add raw decoding later
                    error('Raw decoding is not supported at this time.');
            end
        end
        
        function obj = padsChanged(~, event)
            obj = event.AffectedObject;
            
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
                            % between -5:-3 has already been set to zero, so we'll need to use the
                            % saved image state to try and recover that lost information
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
                                    obj.values((abs(obj.lastpads(side))+1):(abs(obj.lastpads(side))-padUpdate(side)), :) = 0;
                                case 2 % BOT
                                    obj.values((size(obj.values,1)-abs(obj.lastpads(side))+padUpdate(side)+1):end, :) = 0;
                                case 3 % LEFT
                                    obj.values(:, (abs(obj.lastpads(side))+1):(abs(obj.lastpads(side))-padUpdate(side))) = 0;
                                case 4 % RIGHT
                                    obj.values(:, (size(obj.values,2)-abs(obj.lastpads(side))+padUpdate(side)+1):end) = 0;
                            end
                        end
                    end
                end
                
                % create a temporary image where we will place the original image on top 
                % (( current size + (positive padding requested) - (previous padding) ))
                tempImageRows = size(obj.values,1)+max(0,obj.pads(1))+max(0,obj.pads(2))-max(0,obj.lastpads(1))-max(0,obj.lastpads(2));
                tempImageColumns = size(obj.values,2)+max(0,obj.pads(3))+max(0,obj.pads(4))-max(0,obj.lastpads(3))-max(0,obj.lastpads(4));
                tempImage(tempImageRows, tempImageColumns) = 0;
                
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
        
        function obj = mirrorChanged(~, event)
            obj = event.AffectedObject;
            
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
        
        % COMMON FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
        function v = max(obj),      v = max(obj.values(:));                   end
        function v = min(obj),      v = min(obj.values(:));                   end
        
        % MATH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
        % TWO INPUTS
        function obj = plus(a, b),    [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1+arg2; end
        function obj = minus(a, b),   [arg1, arg2, obj] = parseargs(a, b); obj.values = arg1-arg2; end
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

