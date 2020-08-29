
function image_vector = image_reader(subject, serial, resize)
    image_location = ['data\s', int2str(subject), '\', int2str(serial), '.pgm'];
    image = imread(image_location);

    if (resize == 1)
        image = imresize(image, [56 46]);
    end
    
    image = im2double(image);    
    image_vector = image(:);
end
