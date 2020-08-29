
function dec_sorted_eigen_vectors = LDA(X, k, total_classes, total_data_point_per_class)
    total_features = size(X,1);

    M = mean(X,2); %meu
    Sw = zeros(total_features,total_features);
    Sb = zeros(total_features,total_features);

    for i = 1:total_classes
        Xi = X(:, ((i-1)*total_data_point_per_class)+1:i*total_data_point_per_class);
        m_i = mean(Xi,2); %meu_i
        
        for j = 1:total_data_point_per_class
            Sw = Sw + ((Xi(:,j) - m_i) * (Xi(:,j) - m_i)');
        end
        
        Sb = Sb + (total_data_point_per_class * (m_i - M) * (m_i - M)');
    end

    [eigen_vector, eigen_values_matrix] = eig(Sb,Sw);
    eigen_values = diag(eigen_values_matrix);
    
    [unused, sort_order] = sort(eigen_values,'descend');
    dec_sorted_eigen_vectors = eigen_vector(:,sort_order(1:k));
end
