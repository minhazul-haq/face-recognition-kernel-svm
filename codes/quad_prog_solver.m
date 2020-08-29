
function alpha = quad_prog_solver(x, y)
    total_training_data = size(x, 1);

    H = ((1 + x*x').^2).*(y*y');
    f = -ones(total_training_data, 1);
    A = -eye(total_training_data);
    b = zeros(total_training_data, 1);
    Aeq = [y';zeros(total_training_data-1, total_training_data)];
    beq = zeros(total_training_data, 1);
    
    options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'display', 'off');

    alpha = quadprog(H, f, A, b, Aeq, beq, [], [], [], options);
end
