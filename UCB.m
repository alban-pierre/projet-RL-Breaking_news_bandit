function [rew, draws] = UCB(tmax, MAB)

    % Does not work yet since rewards are not bounded by [0,1]
    % Update : I used min and max of rewards in order to put rewards artificially between 0 and 1

    
    NbArms=MAB.nbArms();

    tmax = max(tmax, NbArms);

    mu = zeros(1,NbArms);
    smu = zeros(1,NbArms);
    na = ones(1,NbArms);
    rew = zeros(1,tmax);
    
    for i=1:NbArms
        mu(1,i) = MAB.sample(i);
    end
    rew(1,1:NbArms) = mu;
    smu = mu;

    rmin = min(rew,[],2);
    rmax = max(rew,[],2);

    mu = ((smu./na) - rmin)./(rmax - rmin);
    
    for t=NbArms+1:tmax
        [ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
        rew(1,t) = MAB.sample(ima);
        smu(1,ima) = smu(1,ima) + rew(1,t);
        na(1,ima) = na(1,ima) + 1;
        if (rew(1,t) > rmax)
            rmax = rew(1,t);
            mu = ((smu./na) - rmin)./(rmax - rmin);
        elseif (rew(1,t) < rmin)
            rmin = rew(1,t);
            mu = ((smu./na) - rmin)./(rmax - rmin);
        else
            mu(1,ima) = ((smu(1,ima)/na(1,ima)) - rmin)/(rmax - rmin);
        end
    end
    draws = na;
end
