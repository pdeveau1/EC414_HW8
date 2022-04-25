load kernel-kmeans-2rings.mat
X = data';
k = 2;
Y = k_means(X,k);
scatter(X(1,:),X(2,:),Y);
%% function to calclate k-Means Clustering Algorithm
function y = k_means(X,k)
    [d n] = size(X);
    u = zeros(n,k);
    y = zeros(1,n);
    e = eye(n);
    K = X' * X;
    iteration = 0;
    %initialize: cluster-mean coefficient vectors
    for l = 1:k
        ul = rand(n,1);
        ul = normalize(ul);
        u(:,l) = ul;
    end
    STOP = false;
    %repeat
    while(~STOP)
        %Update labels:
        for j = 1:n
            e_j = e(:,j);
            [~,y(j)] = min((e_j - u)' * K * (e_j - ul));
        end
        %Update cluster-mean coefficient vectors:
        for l = 1:k
            yl = (y == l);
            nl = sum(yl);
            u(:,l) = (1/nl) * sum(e(:,yl),2);
        end
        %check stop conditions
        iteration = iteration + 1;
        %threshold
        %number of iterations
        if(iteration > 100)
            STOP = true;
        end
    end
end