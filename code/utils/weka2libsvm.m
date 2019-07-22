javaaddpath('D:\Program Files\Weka-3-8\weka.jar');
wekaObj = loadARFF('C:\Users\superlj666\Downloads\Compressed\mtr-datasets\mtr-datasets\rf2.arff');
[mdata,featureNames,targetNDX,stringVals,relationName] =  weka2matlab(wekaObj);

if sum(sum(ismissing(mdata)))>0
    fprintf('%s exists missing data\n', char(dataset));
    mdata= rmmissing(mdata);
end

X = mdata(:, 1:576);
y = mdata(:, 577:end);

save('E:\Datasets\rf2.mat', 'X', 'y');