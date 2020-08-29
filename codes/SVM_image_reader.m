
function hog_features = SVM_image_reader(subject, serial)
    image_location = ['data\s', int2str(subject), '\', int2str(serial), '.pgm'];
    image = imread(image_location);
    
    hog_features = hog_feature_vector(image);
end
