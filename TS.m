function [rew, draws] = TS(tmax, MAB)

    % I used min and max of rewards in order to put rewards artificially between 0 and 1
    % It computes only one mean for each arm, but if a hot reward is observed, it pull the expectation of that arm up, thus TS continues to draw that arm
    
    NbArms=MAB.nbArms();

    tmax = max(tmax, NbArms);

    na = ones(1,NbArms);
    rew = zeros(1,tmax);
    sa = zeros(1, NbArms);

    t = 1;
    
    beta = betarnd(sa+1, na-sa+1);
    [ma, ima] = max(beta);
    rew(1,t) = MAB.sample(ima);
    na(1,ima) = na(1,ima) + 1;

    rmin = rew(1,t)*0.999;
    rmax = rew(1,t)*1.001;

    sa(1,ima) = sa(1,ima) + (rew(1,t)-rmin)/(rmax-rmin);
    
    for t=2:tmax
        beta = betarnd(sa+1, na-sa+1);
        [ma, ima] = max(beta);
        rew(1,t) = MAB.sample(ima);
        na(1,ima) = na(1,ima) + 1;
        if (rew(1,t) > rmax)
            sa = sa./((rew(1,t)-rmin)./(rmax-rmin));
            rmax = rew(1,t);
        end
        if (rew(1,t) < rmin)
            sa = 1-(1-sa)./((rmax-rew(1,t))./(rmax-rmin));
            rmin = rew(1,t);
        end
        sa(1,ima) = sa(1,ima) + (rew(1,t)-rmin)/(rmax-rmin);
    end
    draws = na;
end
