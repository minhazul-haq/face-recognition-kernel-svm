
function mean_accuracy = KNN_without_dimension_reduction()
    accuracy = zeros(1, 5);
    indices = (1:10);
    total_classes = 40;
    total_features = 4680;
    total_training_data = 320;
    total_testing_data = 80;
    total_training_data_per_class = 8;
    
    for cross_validation_number = 1:5
        training_indices = indices(1:8);
        testing_indices = indices(9:10);

        index = 1;

        X = ones(total_features, total_training_data);
        
        for subject = 1:total_classes
            for serial = training_indices
                X(:, index) = SVM_image_reader(subject, serial)';
                index = index + 1;
            end
        end

        total_accurate = 0;

        for subject = 1:total_classes
            for serial = testing_indices
                x = SVM_image_reader(subject, serial)';

                matched_index = KNN(X, x);
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
