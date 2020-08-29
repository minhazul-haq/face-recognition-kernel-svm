
%task 1
do_resize = 0;
accuracy_KNN_with_PCA = KNN_with_PCA(do_resize);
disp(['KNN_with_PCA accuracy: ', num2str(accuracy_KNN_with_PCA), '%']);

%task 2
do_resize = 1;
accuracy_KNN_with_PCA_halfly_resized = KNN_with_PCA(do_resize);
disp(['KNN_with_PCA_halfly_resized accuracy: ', num2str(accuracy_KNN_with_PCA_halfly_resized), '%']);

%task 3
apply_PCA_first = 0;
accuracy_KNN_with_LDA = KNN_with_LDA(apply_PCA_first);
disp(['KNN_with_LDA accuracy: ', num2str(accuracy_KNN_with_LDA), '%']);

%task 4
apply_PCA_first = 1;
accuracy_KNN_with_PCA_LDA = KNN_with_LDA(apply_PCA_first);
disp(['KNN_with_PCA_LDA accuracy: ', num2str(accuracy_KNN_with_PCA_LDA), '%']);

%task 5
apply_PCA_first = 0;
accuracy_SVM_with_Kernel = SVM_with_kernel(apply_PCA_first);
disp(['SVM_with_Kernel accuracy: ', num2str(accuracy_SVM_with_Kernel), '%']);

%task 6
apply_PCA_first = 1;
accuracy_SVM_with_PCA_Kernel = SVM_with_kernel(apply_PCA_first);
disp(['SVM_with_PCA_Kernel accuracy: ', num2str(accuracy_SVM_with_PCA_Kernel), '%']);

%experimental task: applying KNN without dimension reduction
accuracy_KNN_without_dimension_reduction = KNN_without_dimension_reduction();
disp(['KNN_without_dimension_reduction accuracy: ', num2str(accuracy_KNN_without_dimension_reduction), '%']);
