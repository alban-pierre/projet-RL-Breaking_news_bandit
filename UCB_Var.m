function [rew, draws, hot_expected, hot_real] = UCB_Var(tmax, MAB)
    %UCB for Breaking News bandit (one hot arm) with inference of variance
    % Update : I used min and max of rewards in order to put rewards artificially between 0 and 1

    
    NbArms=MAB.nbArms();

    tmax = max(tmax, NbArms);

    mu = zeros(1,NbArms);
    smu = zeros(1,NbArms);
    unnorm_rew = zeros(1, NbArms);
    na = ones(1,NbArms);
    rew = zeros(1,tmax);
    
    %Vectors of real hot state and expected hot state
    hot_expected = [];
    hot_real = [];
    
    %disp('----------------init');
    hotState = 0; %Indicator of the state that is supposed hot
    old_na_hot = 0; %Hold the number of pulls before the state became hot
    old_smu_hot = 0;
    old_unnorm_rew = [];
    old_bound = 0;
    
    for i=1:NbArms
        [mu(1,i), h] = MAB.sample(i);
        
        hot_expected = [hot_expected, hotState];
        hot_real = [hot_real, h];
    end
    rew(1,1:NbArms) = mu;
    smu = mu;

    rmin = min(rew,[],2);
    rmax = max(rew,[],2);

    mu = ((smu./na) - rmin)./(rmax - rmin);
    unnorm_rew(1,:) = rew(1,1:NbArms);
    
    for t=NbArms+1:tmax
        %var = ((sum((unnorm_rew-rmin)/(rmax-rmin),1)-mu).^2./na).^(1/2);
        if (hotState == 0) %If no arm is supposed hot
            [ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
            variance = var((unnorm_rew(:,ima)-rmin)/(rmax-rmin));
            [rew(1,t), h] = MAB.sample(ima);
            
            if ((rew(1,t)-rmin)/(rmax-rmin) <= mu(1,ima)+variance+sqrt(1/(2*na(1,ima)))) %The sample is in the range expected: arm not hot
                smu(1,ima) = smu(1,ima) + rew(1,t);
                unnorm_rew(na(1,ima)+1, ima) = rew(1,t);
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
            else %The sampled reward is over the expected range: we suppose it is hot
                hotState = ima;
                %disp(['Hot state detected ', num2str(ima), ' at time ', num2str(t)]);
                %Hold the values before entering hot state
                old_na_hot = na(1,ima);
                old_smu_hot = smu(1,ima);
                old_unnorm_rew = unnorm_rew(:,ima);
                old_bound = mu(1,ima)+variance+sqrt(1/(2*na(1,ima)));
                %Enter hot state with new values
                smu(1,ima) = rew(1,t);
                na(1,ima) = 1;
                unnorm_rew(:,ima) = 0;
                unnorm_rew(1,ima) = rew(1,t);
                %Normalize as usual if needed
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
            hot_expected = [hot_expected, hotState];
            hot_real = [hot_real, h];
        else
            
%             disp(['t:', num2str(t)]);
%             disp(['ima:', num2str(ima)]);
%             disp(['var1:', num2str(sqrt(1/(2*na(1,ima))))]);
%             disp(['var2:', num2str(variance)]);
%             disp(['mu:', num2str(mu(1,ima))]);
            [ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
            [rew(1,t),h] = MAB.sample(ima);
%             disp(['rew:', num2str((rew(1,t)-rmin)/(rmax-rmin))]);
%             disp('unnorm_rew(:,ima):')
%             disp(unnorm_rew(:,ima)');
%             rmin = rmin
%             rmax = rmax
%             disp(' ');
            variance = var((unnorm_rew(:,ima)-rmin)/(rmax-rmin));
            
            if (ima == hotState && (rew(1,t)-rmin)/(rmax-rmin) < mu(1,ima)-variance-sqrt(1./(2*na(1,ima)))) % Leave hot state
                %disp(['Leaving hot state detected ', num2str(ima), ' at time ', num2str(t)]);
                hotState = 0;
                %Take back old values
                na(1,ima) = old_na_hot + 1;
                smu(1,ima) = old_smu_hot + rew(1,t);
                unnorm_rew(:,ima) = 0;
                unnorm_rew(1:size(old_unnorm_rew,1),ima) = old_unnorm_rew;
                unnorm_rew(na(1,ima),ima) = rew(1,t);
                %Normalize if needed
                if (rew(1,t) > rmax)
                    rmax = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                elseif (rew(1,t) < rmin)
                    rmin = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                else
                    mu(1,ima) = ((smu(1,ima)/na(1,ima)) - rmin)/(rmax - rmin);
                end
            %If the two bounds from normal (old) and hot state have an intersection then leave hot state
            elseif (ima == hotState && old_bound > mu(1,ima)-variance)
                %disp(['Leaving hot state detected ', num2str(ima), ' at time ', num2str(t)]);
                hotState = 0;
                %Take back old values
                na(1,ima) = old_na_hot + 1;
                smu(1,ima) = old_smu_hot + rew(1,t);
                unnorm_rew(:,ima) = 0;
                unnorm_rew(1:size(old_unnorm_rew,1),ima) = old_unnorm_rew;
                unnorm_rew(na(1,ima),ima) = rew(1,t);
                %Normalize if needed
                if (rew(1,t) > rmax)
                    rmax = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                elseif (rew(1,t) < rmin)
                    rmin = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                else
                    mu(1,ima) = ((smu(1,ima)/na(1,ima)) - rmin)/(rmax - rmin);
                end
            else %Don't change hot state
                smu(1,ima) = smu(1,ima) + rew(1,t);
                na(1,ima) = na(1,ima) + 1;
                unnorm_rew(na(1,ima)+1, ima) = rew(1,t);
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
            
            
            
            hot_expected = [hot_expected, hotState];
            hot_real = [hot_real, h];
        end
    end
    draws = na;
end
