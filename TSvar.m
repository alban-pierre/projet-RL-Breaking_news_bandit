function [rew, draws] = TSvar(tmax, MAB)

    % Alternative of TS that tries to take into account the variance of rewards to efficiently reduce the exploration step
    % Does not work yet since rewards are not bounded by [0,1]
    
    NbArms=MAB.nbArms();

    tmax = max(tmax, NbArms);

    mu = zeros(1,NbArms);
    smu = zeros(1,NbArms);
    na = ones(1,NbArms);
    rew = zeros(1,tmax);
    sa = zeros(1, NbArms);
    var = zeros(1, NbArms);
    
    for t=1:tmax
        navar = (mu.*(1-mu))./max(10^-10, var);
        navar = max(na, ceil(na.*navar));
        navar = min(navar, na.^2);
        beta = betarnd(sa+1, na-sa+1);
        [ma, ima] = max(beta);
        rew(1,t) = MAB.sample(ima);
        smu(1,ima) = smu(1,ima) + rew(1,t);
        na(1,ima) = na(1,ima) + 1;
        sa(1,ima) = sa(1,ima) + rew(1,t);
        mu(1,ima) = smu(1,ima)/na(1,ima);
        var(1,ima) = ((na(1,ima)-1).*var(1,ima) + (rew(1,t)-mu(1,ima)).^2)./na(1,ima);
    end
    draws = na;
end
