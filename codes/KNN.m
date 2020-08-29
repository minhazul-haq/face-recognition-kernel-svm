
function min_index = KNN(X, x)
    min_distance = Inf;
    min_index = 0;
    
    total_data_points = size(X, 2);
    
    for i = 1:total_data_points
        distance = sum((X(:,i) - x).^2);

        if (distance < min_distance)
            min_distance = distance;
            min_index = i;
        end
    end    
end
