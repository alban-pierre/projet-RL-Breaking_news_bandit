function [mu, Pi, sigma, p_of_k_knowing_X] = em_gaussian(X, mu, Pi, sigma, maxIter, condition_of_stop)

    % Computes the EM algorithm for a mixture of general gaussians

    % Dimensions :
    % D : Dimension of observed points
    % N : Number of points
    % K : Number of gaussians

    % Input : 
    % X       : (D*N)   : Observed points of training set
    % mu      : (D*K)   : Means of the K gaussians
    % Pi      : (1*K)   : Probability of the K gaussians
    % sigma   : (D*D*K) : Covariance of the K gaussians
    % maxIter : Int     : Maximum number of iterations the EM algorithm can make
    
    % Optionnal Input :
    % condition_of_stop : @(mu, Pi, sigma, oldmu, oldPi, oldsigma) -> Bool
                      % : Criterion that decides to stop or to resume iterations of
                        % EM algorithm.

    % Output :
    % mu               : (D*K)   : Means of the K gaussians
    % Pi               : (1*K)   : Probability of the K gaussians
    % sigma            : (D*D*K) : Covariance of the K gaussians
    % p_of_k_knowing_X : (K*N)   : Probability of points X to belong to each gaussian
    
    % Getting sizes
    D = size(X,1);
    N = size(X,2);
    K = size(mu,2);

    % Initialisations
    q = zeros(K,N);
    is_not_converged = 1;
    stooop = 1;
    
    % Construct a reasonable condition stop if there is none yet
    if (nargin < 6)
        x = X(:,randperm(N)(1:min(N,5)));
        avg_dist = sqrt(mean(mean(sqdist(x,x))));
        condition_of_stop = @(x2,x3,x4,x5,x6,x7) (norm([x2(:); x3(:); x4(:)] - ...
                                                       [x5(:); x6(:); x7(:)]) > avg_dist/1000);
    end
    if (nargin < 5)
        maxIter = 1000;
    end

    
    % EM iterations
    while (is_not_converged && (stooop < maxIter))

        % Save varaibles for condition_of_stop variables
        oldmu = mu;
        oldPi = Pi;
        oldsigma = sigma;
        
        % E step (course 3 page 8)
        for k=1:K
            for n=1:N
                q(k,n) = Pi(1,k) * (1/((2*pi)^(D/2)*sqrt(det(sigma(:,:,k))))) * ...
                         exp(-1/2*(X(:,n)-mu(:,k))'*(sigma(:,:,k)^-1)*(X(:,n)-mu(:,k)));
            end
        end
        sq = sum(q,1);
        q = q./repmat(sq,K,1);

        % M step (course 3 page 9-10)
        mu = X*q' ./ repmat(sum(q,2)',D,1);
        Pi = mean(q,2)';
        for k=1:K
            s = 0;
            for n=1:N
                s += (X(:,n)-mu(:,k)) * (X(:,n)-mu(:,k))' * q(k,n);
            end
            sigma(:,:,k) = s/sum(q(k,:),2);
            sigma(:,:,k) = (sigma(:,:,k)+sigma(:,:,k)')/2; % Make sure sigma is symmetric
        end
        
        % Continue the EM algorithm or not
        if (stooop > 1)
            is_not_converged = condition_of_stop(mu, Pi, sigma, oldmu, oldPi, oldsigma);
        end
        stooop++;
    end

    
    if (stooop >= maxIter)
        printf("Warning : Maximum iteration reached.\n");
    end

    
    % Probability of points X to belong to each gaussian
    for k=1:K
        for n=1:N
            q(k,n) = Pi(1,k) * (1/((2*pi)^(D/2)*sqrt(det(sigma(:,:,k))))) * ...
                     exp(-1/2*(X(:,n)-mu(:,k))'*(sigma(:,:,k)^-1)*(X(:,n)-mu(:,k)));
        end
    end
    sq = sum(q,1);
    p_of_k_knowing_X = q./repmat(sq,K,1);

end
