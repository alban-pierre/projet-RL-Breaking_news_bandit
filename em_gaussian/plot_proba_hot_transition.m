function alld = plot_proba_hot_transition(hot_tr, tmax)

    % Computes the probability to be in the normal state accross iterations

    K = size(hot_tr,1);

    alld = zeros(K,tmax);
    alld(1,1) = 1;
    for k=1:K
        pr = ((1:K)' == k);
        for j=1:tmax
            pr = hot_tr*pr;
            alld(k,j) = pr(1,1);
        end
    end
    
    figure;
    plot(alld');

end
