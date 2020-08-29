
function mean_accuracy = KNN_with_PCA(do_resize)
    accuracy = zeros(1, 5);
    indices = (1:10);
    total_testing_data = 80;
    total_training_data_per_class = 8;
    
    for cross_validation_number = 1:5
        training_indices = indices(1:8);
        testing_indices = indices(9:10);

        index = 1;

        if (do_resize == 1)
            X = ones((112/2)*(92/2), 320);
        else
            X = ones(112*92, 320);
        end

        for subject = 1:40
            for serial = training_indices
                X(:, index) = image_reader(subject, serial, do_resize);
                index = index + 1;
            end
        end

        eigen_vectors = PCA(X, 30);
        projected_X = eigen_vectors' * X;

        total_accurate = 0;

        for subject = 1:40
            for serial = testing_indices
                x = image_reader(subject, serial, do_resize);
                projected_x = eigen_vectors' * x; 

                matched_index = KNN(projected_X, projected_x);
                target_subject = ceil(matched_index/total_training_data_per_class);

                if (target_subject == subject)
                    total_accurate = total_accurate + 1;
                end
            end
        end

        accuracy(cross_validation_number) = (total_accurate/total_testing_data)*100;

        indices = circshift(indices, [0,-2]);
    end

    mean_accuracy = mean(accuracy);
end
