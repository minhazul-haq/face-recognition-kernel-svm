
function mean_accuracy = SVM_with_kernel(apply_PCA_first)
    accuracy = zeros(1, 5);
    indices = (1:10);

    for cross_validation_number = 1:5
        training_indices = indices(1:8);
        testing_indices = indices(9:10);
        
        [alpha, b, eigen_vectors, projected_X] = SVM_train(training_indices, apply_PCA_first);
        accuracy(cross_validation_number) = SVM_test(alpha, b, eigen_vectors, projected_X, testing_indices, apply_PCA_first);
        
        indices = circshift(indices, [0,-2]);
    end

    mean_accuracy = mean(accuracy);
end
