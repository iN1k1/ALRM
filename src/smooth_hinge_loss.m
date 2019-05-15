function [ loss ] = smooth_hinge_loss( z )
loss = zeros(length(z),1);
less_or_eq_0 = z <= 0;
between_0_and_1 = z > 0 & z < 1;
loss(less_or_eq_0) = 0.5 - z(less_or_eq_0);
loss(between_0_and_1) = 0.5 * (1 - z(between_0_and_1)).^2;
end

