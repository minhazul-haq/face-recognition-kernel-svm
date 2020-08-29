
function [alpha, b, eigen_vectors, projected_X] = SVM_train(training_image_indices, apply_PCA)
    total_classes = 40;
    total_training_data = 320;
    total_training_data_per_class = 8;
    total_features = 4680;
    index = 1;

    X = zeros(total_training_data, total_features);

    for i = 1:total_classes
        for j = training_image_indices
            X(index,:) = SVM_image_reader(i, j);
            index = index + 1;
        end
    end
    
    if (apply_PCA == 1)
        eigen_vectors = PCA(X', 30);
        projected_X = (eigen_vectors' * X')';
    else
        eigen_vectors = zeros(1,1);
        projected_X = X;
    end
    
    b = zeros(total_classes,1);
    alpha = zeros(total_training_data,total_classes);
    
    %train 40 SVM
    for svm = 1: total_classes
        mini_y = -ones(total_classes,1);
        mini_y(svm, 1) = 1;
        y = kron(mini_y, ones(total_training_data_per_class,1));

        alpha(:,svm) = quad_prog_solver(projected_X, y);
        
        wX = zeros(total_training_data,1);
        
        for i = 1:total_training_data
            for j = 1:total_training_data
                wX(j,1) = wX(j,1) + alpha(i,svm).*y(i).*(kernel(projected_X(i,:), projected_X(j,:)));
            end
        end
        
        for i = 1: total_training_data
            if (alpha(i,svm)~=0)
                b(svm,1) = y(i,1) - wX(i,1);
                break;
            end
        end
    end
end
