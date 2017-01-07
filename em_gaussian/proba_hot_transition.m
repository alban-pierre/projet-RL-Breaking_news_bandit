function ps = proba_hot_transition(hot_tr, p, t)

    % Computes the transition probability to hot states

    K = size(p,1);

    d = 0;
    alld = [0];
    pr = p(:,1);
    for i=2:size(p,2)
        for j=1:t(1,i)
            pr = hot_tr*pr;
        end
        %d = d + 1-sum(pr.*p(:,i),1);
        %alld = [alld, 1-sum(pr.*p(:,i),1)];
        %d = d + sum(abs(pr - p(:,i)),1);
        %alld = [alld, sum(abs(pr - p(:,i)),1)];
        %d = d + sum(abs(pr - p(:,i)),1);
        %alld = [alld, sum(abs(pr - p(:,i)),1)];
        d = d + ((1-pr).^2)'*p(:,i);
        alld = [alld, ((1-pr).^2)'*p(:,i)];
        pr = p(:,i);
    end
    
    ps = d;

end
