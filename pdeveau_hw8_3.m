load kernel-kmeans-2rings.mat
X = data';
k = 2;
Y = k_means(X,k);
K = K_RBF(1000,X);
gscatter(X(1,:),X(2,:),Y);
%% 
function K = K_RBF(n, X)
    sigma = sqrt(0.16);
    K = zeros(n);
    for u = 1:n
        for v = 1:n
            K(u,v) = exp(-(1/(2*(sigma^2))) * (norm(X(:,u)-X(:,v)).^2));
        end
    end
end

%% function to calclate k-Means Clustering Algorithm
function y = k_means(X,k)
    [d n] = size(X);
    u = zeros(n,k);
    y = ones(1,n);
    e = eye(n);
    K = K_RBF(n,X);
    iteration = 0;
    %initialize: cluster-mean coefficient vectors
    for l = 1:k
        ul = rand(n,1);
        ul = ul/norm(ul);
        u(:,l) = ul;
    end
    STOP = false;
    %repeat
    while(~STOP)
        %Update labels:
        for j = 1:n
            e_j = e(:,j);
            if((e_j - u(:,1))' * K * (e_j - u(:,1)) < (e_j - u(:,2))' * K * (e_j - u(:,2)))
                y(j) = 1;
            else
                y(j) = 2;
            end
            %[~,y(j)] = min((e_j - u)' * K * (e_j - u));
        end
        %Update cluster-mean coefficient vectors:
        for l = 1:k
            yl = (y == l);
            nl = sum(yl);
            if nl ~= 0
                sec = 0;
                for j = 1:n
                    sec = sec + (e(:,j) * (y(j) == l));
                end
                u(:,l) = (1/nl) * sec;
                %u(:,l) = (1/nl) * sum(e(:,yl),2);
            else
                u(:,l) = zeros(n,1);
            end
        end
        %check stop conditions
        iteration = iteration + 1;
        %threshold
        %number of iterations
        if(iteration > 100)
            STOP = true;
        end
        gscatter(X(1,:),X(2,:),y);
    end
end