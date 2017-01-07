function [rew, draws] = KNN_UCB(tmax, MAB)

    % Does not work yet since rewards are not bounded by [0,1]

    NbArms=MAB.nbArms();

    tmax = max(tmax, NbArms);

    %mu = zeros(1,NbArms);
    %smu = zeros(1,NbArms);
    na = ones(1,NbArms);
    rew = zeros(1,tmax);
    ta = ones(1,NbArms);
    
    for i=1:NbArms
        mu(1,i) = MAB.sample(i);
        rw{i} = mu(1,i);
        tl{i} = ta(1,i);
        ta(1,i) = 0;
        ta = ta+1;
    end
    rew(1,1:NbArms) = mu;
    smu = mu;
    
    for t=NbArms+1:tmax
        for i=1:NbArms
            mu(1,i) = knn(rw{i}, tl{i}, ta, ceil(sqrt(t)/NbArms));
        end
        [ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
        rew(1,t) = MAB.sample(ima);
        rw{ima} = [rw{ima}, rew(1,t)];
        tl{ima} = [tl{ima}, ta(1,ima)];
        ta(1,ima) = 0;
        ta = ta+1;
        %smu(1,ima)+=rew(1,t);
        na(1,ima) = na(1,ima) + 1;
        %mu(1,ima) = smu(1,ima)/na(1,ima);
    end
    draws = na;
end
