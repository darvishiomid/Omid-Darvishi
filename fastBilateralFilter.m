function filteredImage = fastBilateralFilter(image, sigmaColor, sigmaSpatial)
    [rows, cols, channels] = size(image);
    filteredImage = zeros(rows, cols, channels);

    % محاسبه پارامترهای هسته‌های گاوسی
    kernelSize = 2 * ceil(2 * sigmaSpatial) + 1;
    halfSize = floor(kernelSize / 2);
    [x, y] = meshgrid(-halfSize:halfSize, -halfSize:halfSize);
    spatialKernel = exp(-(x.^2 + y.^2) / (2 * sigmaSpatial^2));
    
    for c = 1:channels
        img = image(:,:,c);
        filtered = zeros(size(img));

        for i = 1:rows
            for j = 1:cols
                iMin = max(i - halfSize, 1);
                iMax = min(i + halfSize, rows);
                jMin = max(j - halfSize, 1);
                jMax = min(j + halfSize, cols);

                region = img(iMin:iMax, jMin:jMax);
                colorKernel = exp(-((region - img(i, j)).^2) / (2 * sigmaColor^2));
                bilateralKernel = spatialKernel((iMin:iMax) - (i - halfSize) + 1, (jMin:jMax) - (j - halfSize) + 1) .* colorKernel;

                filtered(i, j) = sum(sum(bilateralKernel .* region)) / sum(sum(bilateralKernel));
            end
        end

        filteredImage(:,:,c) = filtered;
    end
end
