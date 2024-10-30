% 1. Load and prepare data
cleanImage = imread('test.png');  % Clean image
guideImage = imread('test1.png');  % Guide image

% If the image is single-channel (grayscale), convert it to a three-channel image
if size(cleanImage, 3) == 1
    cleanImage = cat(3, cleanImage, cleanImage, cleanImage);
end

if size(guideImage, 3) == 1
    guideImage = cat(3, guideImage, guideImage, guideImage);
end

% Normalize data
cleanImage = im2double(cleanImage);
guideImage = im2double(guideImage);

% Add Rician noise at 6%
sigma = 0.06;  % Standard deviation of noise (6% of maximum intensity)
n1 = cleanImage + sigma * randn(size(cleanImage));
n2 = sigma * randn(size(cleanImage));
noisyImage = sqrt(n1.^2 + n2.^2);
noisyImage = min(max(noisyImage, 0), 1);  % Clamp values to the range [0, 1]

% Ensure images are three-channel
assert(size(cleanImage, 3) == 3, 'Clean image must be three-channel.');
assert(size(noisyImage, 3) == 3, 'Noisy image must be three-channel.');
assert(size(guideImage, 3) == 3, 'Guide image must be three-channel.');

% 2. Apply Wiener filter to remove noise
filteredImageWiener = zeros(size(noisyImage));
for i = 1:3
    filteredImageWiener(:,:,i) = wiener2(noisyImage(:,:,i), [5 5]);
end

% 3. Set parameters for fast bilateral filter using guide image
sigmaGuide = std2(rgb2gray(guideImage));
alpha = 0.20;
beta = 1.0;
sigmaColor = alpha * (1 / sigmaGuide);
sigmaSpatial = beta * sigmaGuide;

% 4. Implement fast bilateral filter
bilateralFilteredImage = fastBilateralFilter(filteredImageWiener, sigmaColor, sigmaSpatial);

% 5. Apply Unsharp filter to enhance image clarity
unsharpFilteredImage = imsharpen(bilateralFilteredImage, 'Radius', 2, 'Amount', 1.5);

% 6. Define CNN architecture with Residual Connections
layers = [
    imageInputLayer([size(unsharpFilteredImage, 1), size(unsharpFilteredImage, 2), 3], 'Name', 'Input Layer')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'Conv Layer 1')
    batchNormalizationLayer('Name', 'BatchNorm Layer 1')
    reluLayer('Name', 'ReLU Layer 1')
    
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

% Create CNN layer graph
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'ReLU Layer 2', 'Addition Layer 1/in2');  % Connection for Residual Connection

% Display CNN architecture
figure;
plot(lgraph);
title('CNN Architecture with Residual Connections');

% 7. Training settings for CNN
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 16, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 8. Train CNN
[net, info] = trainNetwork(unsharpFilteredImage, cleanImage, lgraph, options);

% Display CNN training progress chart
figure;
plot(info.TrainingLoss, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Training Loss');
title('CNN Training Progress');

% 9. Test and evaluate CNN
denoisedImage = predict(net, unsharpFilteredImage);

% 10. Define autoencoder architecture
autoencoderLayers = [
    imageInputLayer([size(denoisedImage, 1), size(denoisedImage, 2), 3], 'Name', 'Input Layer AE')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'AE Conv Layer 1')
    reluLayer('Name', 'AE ReLU Layer 1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'AE Conv Layer 2')
    reluLayer('Name', 'AE ReLU Layer 2')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'AE Conv Layer 3')
    reluLayer('Name', 'AE ReLU Layer 3')
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'AE Conv Layer 4')
    reluLayer('Name', 'AE ReLU Layer 4')
    transposedConv2dLayer(3, 16, 'Cropping', 'same', 'Name', 'AE Trans Conv Layer 1')
    reluLayer('Name', 'AE ReLU Layer 5')
    transposedConv2dLayer(3, 32, 'Cropping', 'same', 'Name', 'AE Trans Conv Layer 2')
    reluLayer('Name', 'AE ReLU Layer 6')
    transposedConv2dLayer(3, 3, 'Cropping', 'same', 'Name', 'AE Trans Conv Layer 3')
    regressionLayer('Name', 'AE Regression Output')
];

% Create autoencoder layer graph
aeGraph = layerGraph(autoencoderLayers);

% Display autoencoder structure
figure;
plot(aeGraph);
title('Autoencoder Structure');

% 11. Training settings for autoencoder
aeOptions = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 16, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 12. Train autoencoder
[autoencoder, aeInfo] = trainNetwork(denoisedImage, cleanImage, aeGraph, aeOptions);

% Display autoencoder training progress chart
figure;
plot(aeInfo.TrainingLoss, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Training Loss');
title('Autoencoder Training Progress');

% 13. Test and evaluate autoencoder
reconstructedImage = predict(autoencoder, denoisedImage);

% 14. Apply Unsharp filter to the reconstructed image
finalSharpenedImage = imsharpen(reconstructedImage, 'Radius', 2, 'Amount', 1.5);

% 15. Calculate PSNR and SSIM for the final image
finalSharpenedImage = im2double(finalSharpenedImage);
cleanImage = im2double(cleanImage);
psnrValue = psnr(finalSharpenedImage, cleanImage);
ssimValue = ssim(finalSharpenedImage, cleanImage);

% Display PSNR and SSIM results
fprintf('PSNR: %0.4f\n', psnrValue);
fprintf('SSIM: %0.4f\n', ssimValue);

% 16. Draw bar chart for PSNR and SSIM
figure;
bar([psnrValue, ssimValue]);
set(gca, 'xticklabel', {'PSNR', 'SSIM'});
title('Evaluation Metrics for PSNR and SSIM');
ylabel('Value');

% 1. Draw bar chart for CNN weights and biases
cnnLayers = net.Layers;
cnnWeights = [];
cnnBiases = [];

% Extract weights and biases from convolutional layers
for i = 1:length(cnnLayers)
    if isa(cnnLayers(i), 'nnet.cnn.layer.Convolution2DLayer')
        cnnWeights = [cnnWeights; cnnLayers(i).Weights(:)];  % Collect weights
        cnnBiases = [cnnBiases; cnnLayers(i).Bias(:)];  % Collect biases
    end
end

% Display bar chart for CNN weights
figure;
subplot(2, 1, 1);
bar(cnnWeights);
title('CNN Weights');
ylabel('Weight Value');
xlabel('Weight Index');

% Display bar chart for CNN biases
subplot(2, 1, 2);
bar(cnnBiases);
title('CNN Biases');
ylabel('Bias Value');
xlabel('Bias Index');

% 2. Draw bar chart for autoencoder weights and biases
aeLayers = autoencoder.Layers;
aeWeights = [];
aeBiases = [];

% Extract weights and biases from convolutional layers
for i = 1:length(aeLayers)
    if isa(aeLayers(i), 'nnet.cnn.layer.Convolution2DLayer')
        aeWeights = [aeWeights; aeLayers(i).Weights(:)];  % Collect weights
        aeBiases = [aeBiases; aeLayers(i).Bias(:)];  % Collect biases
    end
end

% Display bar chart for autoencoder weights
figure;
subplot(2, 1, 1);
bar(aeWeights);
title('Autoencoder Weights');
ylabel('Weight Value');
xlabel('Weight Index');

% Display bar chart for autoencoder biases
subplot(2, 1, 2);
bar(aeBiases);
title('Autoencoder Biases');
ylabel('Bias Value');
xlabel('Bias Index');
