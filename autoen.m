function autoencoder_with_reference
    % بارگذاری تصویر ورودی
    [fileInput, pathInput] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files (*.png, *.jpg, *.jpeg)'}, 'Select an Input Image');
    if isequal(fileInput, 0)
        disp('User selected Cancel');
        return;
    else
        inputImage = imread(fullfile(pathInput, fileInput));
    end

    % بارگذاری تصویر مرجع
    [fileRef, pathRef] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files (*.png, *.jpg, *.jpeg)'}, 'Select a Reference Image');
    if isequal(fileRef, 0)
        disp('User selected Cancel');
        return;
    else
        referenceImage = imread(fullfile(pathRef, fileRef));
    end

    % تبدیل تصاویر به خاکستری (در صورت نیاز)
    if size(inputImage, 3) == 3
        inputImage = rgb2gray(inputImage);
    end
    if size(referenceImage, 3) == 3
        referenceImage = rgb2gray(referenceImage);
    end

    % تغییر اندازه تصاویر به ابعاد مشخص (مثلاً 512x512)
    inputImage = imresize(inputImage, [512, 512]);
    referenceImage = imresize(referenceImage, [512, 512]);

    % نرمال‌سازی تصاویر به بازه [0, 1]
    inputImage = im2double(inputImage);
    referenceImage = im2double(referenceImage);

    % ایجاد مجموعه داده
    XTrain = reshape(inputImage, [512, 512, 1, 1]); % ورودی
    YTrain = reshape(referenceImage, [512, 512, 1, 1]); % هدف (تصویر مرجع)

    % تعریف مدل اتوانکدر
    autoencoder = initializeAutoencoder();

    % گزینه‌های آموزش
    options = trainingOptions('adam', ...
        'MaxEpochs', 150, ... % تعداد اپوک‌ها
        'MiniBatchSize', 1, ...  % آموزش بر اساس یک تصویر
        'InitialLearnRate', 0.001, ... % نرخ یادگیری
        'Verbose', false, ...
        'Plots', 'training-progress');

    % آموزش مدل
    [autoencoder, info] = trainNetwork(XTrain, YTrain, autoencoder, options);

    % پیش‌بینی تصویر بهبود یافته
    dlYPred = predict(autoencoder, XTrain);
    outputImage = squeeze(dlYPred); % حذف ابعاد تک برای تصویر نهایی
    outputImage = im2uint8(outputImage); % تبدیل به نوع صحیح

    % نمایش تصاویر
    figure;
    subplot(1, 3, 1);
    imshow(inputImage);
    title('Input Image');

    subplot(1, 3, 2);
    imshow(referenceImage);
    title('Reference Image');

    subplot(1, 3, 3);
    imshow(outputImage);
    title('Enhanced Image');
end

function autoencoder = initializeAutoencoder()
    % تعریف لایه‌های اتوانکدر
    layers = [
        imageInputLayer([512 512 1]) % ابعاد ورودی
        convolution2dLayer(3, 16, 'Padding', 'same') % کاهش تعداد فیلترها
        reluLayer()
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 32, 'Padding', 'same') % افزایش تعداد فیلترها
        reluLayer()
        maxPooling2dLayer(2, 'Stride', 2)

        transposedConv2dLayer(2, 32, 'Stride', 2, 'Cropping', 'same'); 
        reluLayer()

        transposedConv2dLayer(2, 16, 'Stride', 2, 'Cropping', 'same'); 
        reluLayer()

        transposedConv2dLayer(3, 1, 'Stride', 1, 'Cropping', 'same'); % اندازه نهایی باید با هدف مطابقت داشته باشد

        regressionLayer()];

    autoencoder = layerGraph(layers);  % ایجاد گراف لایه‌ها
end
