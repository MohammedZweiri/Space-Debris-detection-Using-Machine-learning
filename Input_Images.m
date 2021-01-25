#Authors: Mohammed Zweiri (2020)

close all
clc


 filename = 'Your video name';

[rows, cols, channels, samples] = size(filename);
figure;
for i = 1:samples
    str = sprintf('Sample: %d ', i);  
    imshow(filename(:,:,:,i))
    imwrite(filename(:,:,:,i),['Space_debris_1_' num2str(i) '.jpg'] ,'jpg')
    title(str);
    pause(0.1);
end
