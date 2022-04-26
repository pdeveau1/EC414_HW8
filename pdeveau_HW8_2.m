load kernel-svm-2rings.mat
tmax = 1000;
t = 0:10:tmax;
[d n] = size(x);
%% (a)Create a plot that shows sample-normalized cost
psi_t = SSGD(x,y,tmax);
g_t = [];
for i = 1:length(t)
    g = cost(x,y,psi_t(:,i))
    g_t = [g_t (1/n)*g];
end
figure(1)
plot(t,g_t)
title('How the Sample-Normalized Cost Evolves with Iterations')
xlabel('Iterations');
ylabel('Sample-Normalized Cost');
%% (b)
CCR_t = [];
for i = 1:length(t)
    [r c] = size(y);
    y_pred = zeros(r,c);
    for j = 1:n
       x_test = x(:,j);
       K_xtest = x' * x_test;
       K_xtest_ext = [K_xtest; 1];
       %label predicted by soft-margin SVM
       hSVM = sign(psi_t(:,i)' * K_xtest_ext);
       y_pred(j) = hSVM;
    end
    CCR = sum(y == y_pred) / n;
    CCR_t = [CCR_t CCR];
end
figure(2)
plot(t,CCR_t)
title('How the Training CCR Evolves with Iterations')
xlabel('Iterations');
ylabel('CCR');
%% (c)
fprintf('The training confusion matrix is \n');
disp(confusionmat(y, y_pred));
%% (d)
xpos = x(:,y == 1);
xneg = x(:,y == -1);
xpos_pred = x(:,y_pred == 1);
xneg_pred = x(:,y_pred == -1);
psi = psi_t(:,length(t));

% w = psi(1:n);
% b = psi(n+1);
% syms x_agh
% eqn  = w * x_agh + b == 0
% boundary = solve(eqn)
% boundary = w'*x' + b;

figure(3)
hold on
scatter(xpos(1,:),xpos(2,:),'b+')
scatter(xneg(1,:),xneg(2,:),'r_')
%plot(boundary)
xlabel('x1');
ylabel('x2');
legend('+1','-1')
hold off



figure(4)
hold on
scatter(xpos_pred(1,:),xpos_pred(2,:),'b+')
scatter(xneg_pred(1,:),xneg_pred(2,:),'r_')
xlabel('x1');
ylabel('x2');
legend('+1','-1')
hold off
%% 
function K = K_RBF(n, X)
    sigma = 0.5;
    K = zeros(n);
    for u = 1:n
        for v = 1:n
            K(u,v) = exp(-(1/(2*(sigma^2))) * (norm(X(:,u)-X(:,v)).^2));
        end
    end
end

%% function to find stochastic sub-gradient descent for binary Kernel SVM
function psi_t = SSGD(X,Y,tmax)
    nC = 256;
    [d,n] = size(X);
    psi = zeros(n+1,1);
    K = K_RBF(n,X); %(n x n)
    psi_t = [psi];
    range = 1 : n;
    for t = 1:tmax
        % Choose sample index: j uniformly at random from {1,...,n}
        j = randsample(range, 1);
        y_j = Y(j);
        x_j = X(:,j);
        K_j = K(:,j);
        Kj_ext = [K_j; 1];%(n + 1) x 1

        s_t = 0.256 / t;
        
        % Compute subgradient:
        v = conv2(K,[1,0;0 0]) * psi;

        if(y_j * psi' * Kj_ext < 1)
            v = v - nC * y_j * Kj_ext;
        end
        
        % Update parameters:
        psi = psi - s_t * v;
        if(rem(t, 10) == 0)
            psi_t = [psi_t psi];
        end
    end
end
%% function to calculate the cost
function g = cost(X, Y, psi)
    [d n] = size(X);
    nC = 256;
    C = nC/n;
    K = K_RBF(n,X); %(n x n)
    K_ext = [K; ones(1,n)];

    f0 = (1/2) * psi' * conv2(K,[1,0;0 0]) * psi;
    g = f0;
    for j = 1:n
        fj = C * hinge(Y(j) * psi' * K_ext(:,j));
        g = g + fj;
    end
end
%% hinge function
function out = hinge(t)
    %hinge(t) := max(0, 1 − t)
    if(1 - t > 0)
        out = 1-t;
    else
        out = 0;
    end
end



