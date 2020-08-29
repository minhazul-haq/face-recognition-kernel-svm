
function dec_sorted_eigen_vectors = PCA(X, k)
    [d, N] = size(X);
    A = zeros(d,N);
    average = (1/N) * sum(X,2);
    
    for i = 1:N
        A(:,i) = X(:,i) - average(:);
    end

    covariance = A * A';

    [eigen_vector, eigen_values_matrix] = eig(covariance);
    eigen_values = diag(eigen_values_matrix);

    [unused, sort_order] = sort(eigen_values,'descend');
    dec_sorted_eigen_vectors = eigen_vector(:,sort_order(1:k));
end
