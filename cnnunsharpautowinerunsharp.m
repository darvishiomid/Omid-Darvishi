% 1. بارگذاری و آماده‌سازی داده‌ها
cleanImage = imread('fc.png');  % تصویر تمیز
guideImage = imread('fc1.png');  % تصویر راهنما

% اضافه کردن نویز گوسی با شدت بیشتر
noisyImage = imnoise(cleanImage, 'gaussian', 0, 0.01);

% اضافه کردن نویز نمک و فلفل
noisyImage = imnoise(noisyImage, 'salt & pepper', 0.01);

% نرمالیزه کردن داده‌ها
cleanImage = im2double(cleanImage);
noisyImage = im2double(noisyImage);
guideImage = im2double(guideImage);

% اطمینان از اینکه تصاویر سه کانالی هستند
assert(size(cleanImage, 3) == 3, 'تصویر تمیز باید سه کانالی باشد.');
assert(size(noisyImage, 3) == 3, 'تصویر نویزدار باید سه کانالی باشد.');
assert(size(guideImage, 3) == 3, 'تصویر راهنما باید سه کانالی باشد.');

% 2. اعمال فیلتر Wiener برای حذف نویز و حفظ جزئیات
filteredImageWiener = zeros(size(noisyImage));
for i = 1:3
    filteredImageWiener(:,:,i) = wiener2(noisyImage(:,:,i), [5 5]);
end

% نمایش تصویر پس از فیلتر Wiener
figure;
subplot(1, 4, 1); imshow(cleanImage); title('تصویر تمیز');
subplot(1, 4, 2); imshow(noisyImage); title('تصویر نویزدار با شدت بیشتر');
subplot(1, 4, 3); imshow(filteredImageWiener); title('تصویر پس از فیلتر Wiener');

% 3. تنظیم پارامترهای فیلتر دوطرفه سریع با استفاده از تصویر راهنما
sigmaGuide = std2(rgb2gray(guideImage));
alpha = 0.20;  
beta = 1.0;

sigmaColor = alpha * (1 / sigmaGuide);
sigmaSpatial = beta * sigmaGuide;

% 4. پیاده‌سازی فیلتر دوطرفه سریع
bilateralFilteredImage = fastBilateralFilter(filteredImageWiener, sigmaColor, sigmaSpatial);

% 5. اعمال فیلتر Unsharp برای افزایش وضوح تصویر پس از فیلتر دوطرفه
unsharpFilteredImage = imsharpen(bilateralFilteredImage, 'Radius', 2, 'Amount', 1.5);

% نمایش تصویر پس از فیلتر Unsharp
figure;
subplot(1, 4, 1); imshow(cleanImage); title('تصویر تمیز');
subplot(1, 4, 2); imshow(noisyImage); title('تصویر نویزدار با شدت بیشتر');
subplot(1, 4, 3); imshow(bilateralFilteredImage); title('تصویر پس از فیلتر دوطرفه');
subplot(1, 4, 4); imshow(unsharpFilteredImage); title('تصویر پس از فیلتر Unsharp');

% 6. تعریف معماری شبکه CNN با Residual Connections
layers = [
    imageInputLayer([size(unsharpFilteredImage, 1), size(unsharpFilteredImage, 2), 3], 'Name', 'Input Layer')  % تعداد کانال‌ها 3
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'Conv Layer 1')
    batchNormalizationLayer('Name', 'BatchNorm Layer 1')
    reluLayer('Name', 'ReLU Layer 1')
    
    % اولین Residual Block
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'Conv Layer 2')
    batchNormalizationLayer('Name', 'BatchNorm Layer 2')
    reluLayer('Name', 'ReLU Layer 2')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'Conv Layer 3')
    batchNormalizationLayer('Name', 'BatchNorm Layer 3')
    
    additionLayer(2, 'Name', 'Addition Layer 1')
    reluLayer('Name', 'ReLU Layer 3')
    
    dropoutLayer(0.5, 'Name', 'Dropout Layer')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'Conv Layer 4')
    batchNormalizationLayer('Name', 'BatchNorm Layer 4')
    reluLayer('Name', 'ReLU Layer 4')
    
    convolution2dLayer(3, 3, 'Padding', 'same', 'Name', 'Final Conv Layer')
    
    regressionLayer('Name', 'Regression Output')
];

% ایجاد گراف شبکه
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'ReLU Layer 2', 'Addition Layer 1/in2');  % اتصال برای Residual Connection

% 7. تنظیمات آموزش
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 16, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 8. آموزش شبکه
[net, info] = trainNetwork(unsharpFilteredImage, cleanImage, lgraph, options);

% 9. تست و ارزیابی
denoisedImage = predict(net, unsharpFilteredImage);

% 10. نمایش تصاویر
figure;
subplot(1, 4, 1); imshow(cleanImage); title('تصویر تمیز');
subplot(1, 4, 2); imshow(noisyImage); title('تصویر نویزدار با شدت بیشتر');
subplot(1, 4, 3); imshow(unsharpFilteredImage); title('تصویر پس از فیلتر دوطرفه و Unsharp');
subplot(1, 4, 4); imshow(denoisedImage); title('تصویر بدون نویز توسط CNN با Residual Connections');

% 11. تعریف اتوانکدر
autoencoderLayers = [
    imageInputLayer([size(denoisedImage, 1), size(denoisedImage, 2), 3], 'Name', 'Input Layer AE')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'AE Conv Layer 1')
    reluLayer('Name', 'AE ReLU Layer 1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'AE Conv Layer 2')
    reluLayer('Name', 'AE ReLU Layer 2')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'AE Conv Layer 3')
    reluLayer('Name', 'AE ReLU Layer 3')
    
    % لایه فشرده‌سازی
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'AE Conv Layer 4')
    reluLayer('Name', 'AE ReLU Layer 4')
    
    % لایه بازسازی
    transposedConv2dLayer(3, 16, 'Cropping', 'same', 'Name', 'AE Trans Conv Layer 1')
    reluLayer('Name', 'AE ReLU Layer 5')
    transposedConv2dLayer(3, 32, 'Cropping', 'same', 'Name', 'AE Trans Conv Layer 2')
    reluLayer('Name', 'AE ReLU Layer 6')
    transposedConv2dLayer(3, 3, 'Cropping', 'same', 'Name', 'AE Trans Conv Layer 3')
    
    regressionLayer('Name', 'AE Regression Output')
];

% ایجاد گراف اتوانکدر
aeGraph = layerGraph(autoencoderLayers);

% 12. تنظیمات آموزش اتوانکدر
aeOptions = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 16, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 13. آموزش اتوانکدر
[autoencoder, aeInfo] = trainNetwork(denoisedImage, cleanImage, aeGraph, aeOptions);

% 14. تست و ارزیابی اتوانکدر
reconstructedImage = predict(autoencoder, denoisedImage);

% 15. اعمال فیلتر Unsharp به تصویر بازسازی شده
finalSharpenedImage = imsharpen(reconstructedImage, 'Radius', 2, 'Amount', 1.5);

% 16. نمایش تصاویر
figure;
subplot(1, 4, 1); imshow(cleanImage); title('تصویر تمیز');
subplot(1, 4, 2); imshow(denoisedImage); title('تصویر بدون نویز توسط CNN');
subplot(1, 4, 3); imshow(reconstructedImage); title('تصویر بازسازی شده توسط اتوانکدر');
subplot(1, 4, 4); imshow(finalSharpenedImage); title('تصویر نهایی پس از فیلتر Unsharp');
