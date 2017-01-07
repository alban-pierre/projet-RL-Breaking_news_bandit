function ps = proba_hot_transition(hot_tr, p, t)

    % Computes the transition probability to hot states

    K = size(p,1);

    d = 0;
    pr = ((1:K)' == 1);
    for i=1:size(p,2)
        for j=1:t
            pr = hot_tr*pr;
        end
        d = d + 1-sum(pr.*p(:,i),1);
        pr = p(:,i);
    end
    
    ps = d;

end
