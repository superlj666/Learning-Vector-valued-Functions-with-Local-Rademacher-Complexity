javaaddpath('D:\Program Files\Weka-3-8\weka.jar');
wekaObj = loadARFF('C:\Users\superlj666\Downloads\Compressed\scene\scene.arff');
[mdata,featureNames,targetNDX,stringVals,relationName] =  weka2matlab(wekaObj);

X = mdata(:, 1:294);
y = mdata(:, 295:end);

save('E:\Datasets\scene.mat', 'X', 'y');