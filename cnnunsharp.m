% 1. بارگذاری و آماده‌سازی داده‌ها
cleanImage = imread('clean_medical_image.png');  % تصویر تمیز
guideImage = imread('guide_image.png');  % تصویر راهنما

% اضافه کردن نویز گوسی با شدت بیشتر
noisyImage = imnoise(cleanImage, 'gaussian', 0, 0.05);

% اضافه کردن نویز نمک و فلفل
noisyImage = imnoise(noisyImage, 'salt & pepper', 0.02);

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
subplot(1, 5, 1); imshow(cleanImage); title('تصویر تمیز');
subplot(1, 5, 2); imshow(noisyImage); title('تصویر نویزدار با شدت بیشتر');
subplot(1, 5, 3); imshow(filteredImageWiener); title('تصویر پس از فیلتر Wiener');

% 3. تنظیم پارامترهای فیلتر دوطرفه سریع با استفاده از تصویر راهنما
sigmaGuide = std2(rgb2gray(guideImage));
alpha = 0.20;  
beta = 1.0;

sigmaColor = alpha * (1 / sigmaGuide);
sigmaSpatial = beta * sigmaGuide;

% 4. پیاده‌سازی فیلتر دوطرفه سریع
bilateralFilteredImage = fastBilateralFilter(filteredImageWiener, sigmaColor, sigmaSpatial);

% نمایش تصویر پس از فیلتر دوطرفه سریع
subplot(1, 5, 4); imshow(bilateralFilteredImage); title('تصویر پس از فیلتر دوطرفه سریع');

% 5. اعمال فیلتر Unsharp برای بازگردانی جزئیات
unsharpImage = imsharpen(bilateralFilteredImage, 'Radius', 2, 'Amount', 1.5);

% نمایش تصویر پس از فیلتر Unsharp
subplot(1, 5, 5); imshow(unsharpImage); title('تصویر پس از فیلتر Unsharp');

% 6. تعریف معماری شبکه CNN با Residual Connections
layers = [
    imageInputLayer([size(unsharpImage, 1), size(unsharpImage, 2), 3], 'Name', 'Input Layer')  % تعداد کانال‌ها 3
    
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
    'MaxEpochs', 100, ...
    'MiniBatchSize', 16, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 8. آموزش شبکه
[net, info] = trainNetwork(unsharpImage, cleanImage, lgraph, options);

% 9. تست و ارزیابی
denoisedImage = predict(net, unsharpImage);

% 10. نمایش تصاویر
figure;
subplot(1, 5, 1); imshow(cleanImage); title('تصویر تمیز');
subplot(1, 5, 2); imshow(noisyImage); title('تصویر نویزدار با شدت بیشتر');
subplot(1, 5, 3); imshow(bilateralFilteredImage); title('تصویر پس از فیلتر دوطرفه');
subplot(1, 5, 4); imshow(unsharpImage); title('تصویر پس از فیلتر Unsharp');
subplot(1, 5, 5); imshow(denoisedImage); title('تصویر بدون نویز توسط CNN با Residual Connections');
saveas(fig, 'image_comparison_residual_unsharp.png');

% 11. تحلیل ساختار شبکه
analyzeNetwork(net);

% 12. رسم نمودار Training Loss
figure;
plot(info.TrainingLoss, 'LineWidth', 2);
grid on;
xlabel('Epoch');
ylabel('Training Loss');
title('نمودار Training Loss در طول آموزش', 'FontSize', 14);
