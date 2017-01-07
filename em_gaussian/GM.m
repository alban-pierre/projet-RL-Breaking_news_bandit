function [rew, draws] = GM(tmax, MAB)

    % Computes the exact model probabilities
    
    NbArms=MAB.nbArms();

    nmin = 5;
    K = 2;
    tmax = max(tmax, NbArms*K*(nmin+1));
    rew = zeros(1,tmax);
    draws = zeros(1,tmax);


    clear t;
    clear rw;
    clear ps;
    for j=1:NbArms
        t{j} = [];
        rw{j} = [];
        ps{j} = ones(K)/K;
    end

    ta = ones(1,NbArms);

    tt = 1;
    for i=1:(nmin+1)*K
        for j=randperm(NbArms)
            r = MAB.sample(j);
            rew(1,tt) = r;
            draws(1,tt) = j;
            t{j} = [t{j}, ta(1,j)];
            rw{j} = [rw{j}, r];
            ta(1,j) = 0;
            ta = ta + 1;
            tt = tt + 1;
        end
    end

    while (tt < tmax)
        
        clear means;
        clear sigmas;
        clear p;
        best = zeros(1,NbArms);
        for j=1:NbArms
            [means{j}, sigmas{j}, p{j}] = em_approx(rw{j}, K, nmin);
            ps{j} = hot_transitions(p{j}, t{j}, ps{j});
            p_hot = p{j}(:,end);
            for i=1:ta(1,j);
                p_hot = ps{j}*p_hot;
            end
            best(1,j) = means{j}*p_hot;
        end

        [~, j] = max(best);
        if (rand(1) < 0.1)
            j = randi(NbArms);
        end

        r = MAB.sample(j);
        rew(1,tt) = r;
        draws(1,tt) = j;
        t{j} = [t{j}, ta(1,j)];
        rw{j} = [rw{j}, r];
        ta(1,j) = 0;
        ta = ta + 1;
        tt = tt + 1;
    end
    
end

