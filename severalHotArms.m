classdef severalHotArms<handle

    % Class that gather many different arms for different states

    % It allows several classes to be hot (state > 1) at the same time
    % The status (hot or not) of all arms is updated when one arm is sampled

    % Input :
    % istate : (1*N)   : Initial state of each arm
    % type   : (S*N)   : Type of each arm (gaussian only for now)
    % mean   : (S*N)   : Mean of each arm
    % v      : (S*N)   : Variance of each arm
    % p      : (S*S*N) : Transition probabilities of each arm
    
    properties
        h    % (1*N)   : Which state is hot
        p    % (S*S*N) : Transition probabilities
        arms % (S*N)   : Each arm
    end
    
    methods
        function self = severalHotArms(istate, type, mean, v, p)
            self.h=istate;
            for n=1:size(type,2)
                for s=1:size(type,1)
                    switch type(s,n)
                        case 1 % gaussian
                            self.arms{s,n} = armGaussian(mean(s,n), v(s, n));
                        otherwise
                            self.arms{s,n}.sample = @() assert(false, 'Reached undefined state');
                    end
                end
            end
            self.p = p;
        end
        
        function [reward, hotState] = sample(self, s)
            [maxh, hotState] = max(self.h,[],2);
            if (maxh == 1)
                hotState = 0;
            end
            reward = self.arms{self.h(1,s), s}.sample();
            for n=1:size(self.h,2)
                self.h(1,n) = mnrnd(1,self.p(self.h(1,n),:,n)) * (1:size(self.p,1))';
            end
        end

        function NbArms = nbArms(self)
            NbArms = size(self.h,2);
        end
        
    end
end
