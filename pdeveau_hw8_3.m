load kernel-kmeans-2rings.mat
X = data';
k = 2;
Y = k_means_Kernel(X,k);
figure(1)
gscatter(X(1,:),X(2,:),Y)
title('Cluster after Kernel K-means')
xlabel('x1')
ylabel('x2')
%% Use the RBF kernel with σ^2 = 0.16
function K = RBF_kernel(X)
    [d,n] = size(X);
    sigma = sqrt(0.16);
    K = zeros(n);
    first = -1/(2*(sigma^2));
    for u = 1:n
        for v = 1:n
            sec = (norm(X(:,u) - X(:,v))).^2;
            K(u,v) = exp(first * sec);
        end
    end
end
%% WCSS
function wcss = WCSS(e, alpha, K, y)
    [~,n] = size(K);
    wcss = 0;
    for j = 1:n
        wcss = wcss + (e(:,j)-alpha(:,y(j)))' * K * (e(:,j)-alpha(:,y(j)));
    end
end
%% 
function y = k_means_Kernel(X,k)
    [d,n] = size(X);
    y = ones(1,n);
    u = zeros(n,k);
    e = eye(n);
    K = RBF_kernel(X);
    iteration = 0;
    wcss_old = 0; wcss_new = 0;
    %initialize: cluster-mean coefficient vectors u` ∈ Rn, ` = 1, . . . , k, STOP = FALSE
    for l = 1:k
        %choose all coordinates if u1, u2 uniformly at random between 0 and 1
        ul = rand(n,1);
        %scale them so that they have uniy 1-norm
        u(:,l) = ul/norm(ul);
    end
    STOP = false;

    %repeat
    while(~STOP)
        iteration = iteration + 1;
        %update labels:
        for j = 1:n
            check1 = (e(:,j) - u(:,1))' * K * (e(:,j) - u(:,1));
            check2 = (e(:,j) - u(:,2))' * K * (e(:,j) - u(:,2));
            if(check1 < check2)
                y(j) = 1;
            else
                y(j) = 2;
            end
        end
        %update cluster-mean coefficient vectors:
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
        %check stop
        if(iteration > 50)
            STOP = true;
        end
        wccs_old = wcss_new;
        wcss_new = WCSS(e, u, K, y);
        if(wcss_old == wcss_new)
            %STOP = true;
        end
        
    end
end

