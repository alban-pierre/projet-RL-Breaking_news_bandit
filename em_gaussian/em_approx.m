function [means, sigmas, p] = em_approx(X, K, nmin)

    if (nargin < 3)
        nmin = 5;
    end

    % Dimensions
    D = size(X,1);
    N = size(X,2);

    if (N <= 0)
        means = zeros(D,K);
        sigmas = repmat(eye(D), [1,1,K]);
        p = zeros(K,0);
    elseif (N <= K)
        k = repmat(X, 1, ceil(K/N));
        means = k(:,1:K);
        sigmas = repmat(eye(D), [1,1,K]);
        p = ones(K,N)/K;
    elseif (N <= nmin*K)
        % K-means initialisation
        [k, allk] = kmeans(X,K);
        dd = sqdist(k,X);
        [~,imin] = min(dd,[],1);        
        for kk=1:K
            Pi(1,kk) = sum(imin==kk)/N;
            means(:,kk) = mean(X(:,imin==kk),2);
            sigmas(:,:,kk) = cov(X(:,imin==kk)');
        end
        p = (imin == (1:K)');
    else
        % K-means initialisation
        [k, allk] = kmeans(X,K);
        
        % EM of mixture gaussians

        % Initialisation
        dd = sqdist(k,X);
        [~,imin] = min(dd,[],1);        
        for kk=1:K
            Pi(1,kk) = sum(imin==kk)/N;
            mu(:,kk) = mean(X(:,imin==kk),2);
            sigma(:,:,kk) = cov(X(:,imin==kk)');
        end
        
        % EM
        [means, Pi_g, sigmas, p] = em_gaussian(X, mu, Pi, sigma);
    end

end

        
    
    
