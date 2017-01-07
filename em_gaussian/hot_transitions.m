function ps = hot_transitions(p, t, ps_init)

    % Computes the transition probability to hot states via a gradient descent with a fixed number of iterations

    % p : (K*N) : probabilities to belong to each state
    % t : (1*N) : time between 2 consecutive samplings of p (min 1)
    % ps_init : (K*K) : initial transition probabilities
    
    K = size(p,1);
    
    if (nargin < 3)
        ps_init = ones(K)/K;
    end
    
    ps = ps_init;
    assert(all(abs(sum(ps,1) - ones(1,K)) < 0.000001), 'Wrong ps_init');
    
    clear dif;
    eps = 0.00001;
    i = 1;
    for k = 1:K
        for k1 = 1:K
            for k2 = k1+1:K
                dif{i} = zeros(K);
                dif{i}(k1,k) = eps;
                dif{i}(k2,k) = -eps;
                i = i+1;
            end
            %dif{i} = zeros(K);
            %dif{i}(k1,k) = eps;
            %i = i+1;
        end
    end

    depl = 1;

    for i=1:100
        p_hot = proba_hot_transition(ps, p, t);
        for i=1:size(dif,2)
            p_dif(1,i) = proba_hot_transition(ps+dif{i}, p, t);
        end
        p_dif = p_hot - p_dif;
        %p_dif = reshape(p_dif, K, K);
        %p_dif = p_dif - repmat(mean(p_dif,1),K,1);
        %p_dif = reshape(p_dif, 1, K*K);
        p_dif = p_dif ./ norm(p_dif);
        for i=1:size(dif,2)
            ps = ps + 100*p_dif(1,i)*dif{i};
        end
        % Verification that we are still between 0 and 1
        ps = max(ps, eps);
        ps = min(ps, 1-eps);
        %ps(1,:) = 1-sum(ps(2:end,:), 1);
        ps = ps ./ sum(ps,1);
    end

end
