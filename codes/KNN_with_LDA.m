
function mean_accuracy = KNN_with_LDA(apply_PCA_first)
    accuracy = zeros(1, 5);
    indices = (1:10);
    total_training_data = 320;
    total_testing_data = 80;
    total_classes = 40;
    total_training_data_per_class = 8;

    for cross_validation_number = 1:5
        training_indices = indices(1:8);
        testing_indices = indices(9:10);

        index = 1;

        X = ones(112*92, 320);
        
        for subject = 1:40
            for serial = training_indices
                image_vector = image_reader(subject, serial, 0);
                X(:, index) = image_vector;
                index = index + 1;
            end
        end
        
        if (apply_PCA_first == 1)
            eigen_vectors_PCA = PCA(X, total_training_data);
            pca_projected_X = eigen_vectors_PCA' * X;

            eigenVectorsLDA = LDA(pca_projected_X, 30, total_classes, total_training_data_per_class);
            lda_projected_X = eigenVectorsLDA' * pca_projected_X;
        else
            eigenVectorsLDA = LDA(X, 30, total_classes, total_training_data_per_class);
            lda_projected_X = eigenVectorsLDA' * X;
        end
        
        total_accurate = 0;

        for subject = 1:40
            for serial = testing_indices
                image_vector = image_reader(subject, serial, 0);
               
                if (apply_PCA_first == 1)
                    lda_projected_x = eigenVectorsLDA' * (eigen_vectors_PCA' * image_vector);
                else
                    lda_projected_x = eigenVectorsLDA' * image_vector;
                end
                
                matched_index = KNN(lda_projected_X, lda_projected_x);
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
