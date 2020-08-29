
function accuracy = SVM_test(alpha, b, eigen_vectors, projected_X, testing_image_indices, apply_PCA)
    total_classes = 40;
    total_training_data = 320;
    total_training_data_per_class = 8;
    total_testing_data = 80;
    total_features = 4680;
    index = 1;

    Z = zeros(total_testing_data, total_features);
    
    for i = 1:total_classes
        for j = testing_image_indices
            Z(index,:) = SVM_image_reader(i, j);
            index = index + 1;
        end
    end

    if (apply_PCA == 1)
        projected_Z = (eigen_vectors' * Z')';
    else
        projected_Z = Z;
    end
    
    accurate = 0;

    for i = 1:total_testing_data
        wz = zeros(total_classes,1);

        for svm = 1:total_classes        
            mini_y = -ones(total_classes,1);
            mini_y(svm, 1) = 1;
            y = kron(mini_y, ones(total_training_data_per_class,1));

            wz_elem = 0;
            
            for j=1:total_training_data
                wz_elem = wz_elem + alpha(j,svm).*y(j).*(kernel(projected_X(j,:), projected_Z(i,:)));
            end
            
            wz(svm,:) = wz_elem;
        end
                
        result = wz + b;
                
        [maxVal, maxIndex] = max(result);

        if (maxIndex == ceil(i/2))
            accurate = accurate + 1;
        end
    end
    
    accuracy = (accurate / total_testing_data)*100;
end
