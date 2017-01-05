function [rew, draws] = TS(tmax, MAB)

    NbArms=length(MAB);

    tmax = max(tmax, NbArms);

    mu = zeros(1,NbArms);
    na = ones(1,NbArms);
    rew = zeros(1,tmax);
    sa = zeros(1, NbArms);
    
    for t=1:tmax
	beta = betarnd(sa+1, na-sa+1);
	[ma, ima] = max(beta);
	rew(1,t) = sample(MAB{ima});
	na(1,ima)++;
	sa(1,ima)+=rew(1,t);
    end
    draws = na;
end
