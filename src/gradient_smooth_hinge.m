function [grad] = gradient_smooth_hinge(score, weight, y, x_diff_P, x_diff)

leq_zero = score <= 0;
btw_zero_one = score > 0 & score < 1;

w = repmat( (weight(leq_zero) .* y(leq_zero))', size(x_diff_P,1), 1);
if  ~isempty(w)
    grad = (w .* x_diff_P(:, leq_zero)) * x_diff(:,leq_zero)';
else
    grad = 0;
end

w = repmat( (weight(btw_zero_one) .* y(btw_zero_one) .* (1 - score(btw_zero_one)) )', size(x_diff_P,1), 1);
if  ~isempty(w)
    grad = grad + ((w .* x_diff_P(:, btw_zero_one)) * x_diff(:,btw_zero_one)');
end

end