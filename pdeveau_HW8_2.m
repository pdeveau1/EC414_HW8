load kernel-svm-2rings.mat
[d,n] = size(x);
K = RBF_kernel(x);
%% (a) Create a plot which shows how the sample-normalized cost 1/n g(ψ) evolves with iteration number t 
psi_t = SSGD_kernel(x,y);
g_t = zeros(1,100);
for t = 1:100
    g_t(t) = (1/n) * cost(psi_t(:,t), K, y);
end

figure(1)
plot(10:10:1000,g_t)
title('How the Sample-Normalized Cost Evolves with Iteration')
xlabel('Iteration')
ylabel('Sample-Normalized Cost')
%% (b)
CCR_t = zeros(1,100);
[r c] = size(y);
for t = 1:100
    y_pred = zeros(r,c);
    for test = 1:n
        K_test = K(:,test);
        y_pred(test) = h_svm(psi_t(:,t),K_test);
    end
    CCR_t(t) = (sum(y == y_pred)) / n;
end
figure(2)
plot(10:10:1000,CCR_t)
title('How the training CCR evolves with iteration number')
xlabel('Iteration')
ylabel('CCR')
%% (c)
fprintf('The final training confusion matrix is\n');
disp(confusionmat(y,y_pred))
%% (d)
xpos = x(:,y == 1);
xneg = x(:,y == -1);
xpos_pred = x(:,y_pred == 1);
xneg_pred = x(:,y_pred == -1);
boundary = [];
for i = 1:n-1
    if(y_pred(i) ~= y_pred(i+1))
        boundary = [boundary x(:,i) x(:,i)];
    end
end
figure(3)
hold on
scatter(xpos(1,:),xpos(2,:),'b+')
scatter(xneg(1,:),xneg(2,:),'r_')
scatter(boundary(1,:),boundary(2,:),'filled','go')
%plot(boundary)
title('Training set and decision boundary')
xlabel('x1');
ylabel('x2');
legend('+1','-1','boundary')
hold off


%% 
function K = RBF_kernel(X)
    sigma = 0.5;
    [~,n] = size(X);
    K = zeros(n);
    for u = 1:n
        for v = 1:n
            first = -1/(2*(sigma^2));
            second = (norm(X(:,u) - X(:,v)))^2;
            K(u,v) = exp(first*second);
        end
    end
end
%% 
function psi_t = SSGD_kernel(X,Y)
    [d,n] = size(X);
    tmax = 1000;
    nC = 256;
    K = RBF_kernel(X);
    psi_t = zeros(n+1,100)
    %initialize
    psi = zeros(n+1,1);
    for t = 1:tmax
        s_t = 0.256/t;
        %choose sample index:
        j = randi(n,1);
        yj = Y(j);
        Kj_ext = [K(:,j);1];
        %compute subgradient
        v = conv2(K,[1,0;0,0]) * psi;
        if(yj*psi'*Kj_ext < 1)
            v = v - nC * yj * Kj_ext;
        end
        %update parameters:
        psi = psi - s_t * v;
        if(rem(t,10) == 0)
            psi_t(:,t/10) = psi;
        end
    end
end
%% 
function g = cost(psi, K, y)
    C = 256;
    n = length(y);
    f0 = (1/2) * psi' * conv2(K,[1,0;0,0]) * psi;
    g = f0;
    for j = 1:n
        f = C * hinge(y(j) * psi'* [K(:,j);1]);
        g = g + f;
    end
end
%% 
function out = hinge(t)
    if(0 > 1 - t)
        out = 0;
    else
        out = 1-t;
    end
end
%%
function label = h_svm(psi, K)
    label = sign(psi' * [K;1]);
end

