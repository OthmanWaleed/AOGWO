%--------------------------------------------------------------------------
% Title: An Adaptive Oppositional Grey Wolf Optimizer for Complex Engineering Problems
% Authors: Othman Waleed Khalid, Nor Ashidi Mat Isa, Karrar Mohsin Alwan, 
%          Sew Sun Tiang, Ahbishek Sharma, Tarek Berghout, Jun-Jiat Tiang, Wei Hong Lim
% Year: 2026
%
% Description: 
% This script runs the AOGWO algorithm against the CEC-2017 benchmark suite. 
% As specified in standard benchmark methodologies, it performs 30 independent 
% runs for each test function to calculate the robust Mean and Standard Deviation 
% of the final fitness values. F2 is automatically skipped. It displays the 
% statistical results in the command window and plots the average convergence 
% curve (using a red diamond marker) over the 30 runs.
%--------------------------------------------------------------------------
clear, clc, close all;  

% ---------------- Test Setup ----------------
dimension = 50; % Spatial dimension (D) required for testing   
maxIter = 200;   % Maximum number of iterations  
maxFES = 100000; % Maximum number of function evaluations  
numRuns = 30;    % Number of independent runs for statistical robustness [Strength point from paper]

disp('===========================================================');
disp('Starting AOGWO Evaluation on CEC2017 Benchmark Suite...');
disp('===========================================================');

% Iterate through all 30 test functions 
for test_function_index = 1:30  
    
    % F2 is officially deleted from the CEC2017 suite
    if test_function_index == 2  
        disp('>> F2 has been deleted from the suite. Skipping...');
        continue; 
    end  

    % Arrays to store the final fitness and convergence curves for the 30 runs
    best_fitness_runs = zeros(1, numRuns);
    convergence_curves_runs = zeros(numRuns, maxIter);

    % ---------------- 30 Independent Runs Loop ----------------
    for run = 1:numRuns
        % Package options for AOGWO
        opts = struct();  
        opts.maxIter = maxIter;  
        opts.maxFES = maxFES;  
        opts.dimension = dimension;  
        opts.testFunction = test_function_index;  

        % Run the Adaptive Oppositional Grey Wolf Optimizer
        result = AOGWO_17(opts);  

        % Store the best fitness obtained at the end of this run
        best_fitness_runs(run) = result.bestFitness;
        
        % Store the convergence curve of this run
        convergence_curves_runs(run, :) = result.convergenceCurve;
    end

    % ---------------- Statistical Calculations ----------------
    % Calculate Mean and Standard Deviation over the 30 independent runs
    current_mean_fitness = mean(best_fitness_runs);  
    current_std_dev_fitness = std(best_fitness_runs);  
    
    % Calculate the average convergence curve over the 30 runs for plotting
    avg_convergence_curve = mean(convergence_curves_runs, 1);

    % Display numerical results instantly in the command window
    fprintf('F%d | Mean Fitness: %e | Std Dev: %e\n', test_function_index, current_mean_fitness, current_std_dev_fitness);

    % ---------------- Plotting Average Convergence Curve ----------------
    figure('Name', ['Average Convergence Curve - F', num2str(test_function_index)]);  
    hold on;  

    % Plotting exact data points iteration by iteration
    x_values = 1:maxIter;  
    
    % Using a red diamond marker (-rd) customized to plot clearly across 200 iter
    % Utilizing MarkerIndices spaced every 10 iterations to prevent clutter
    plot(x_values, avg_convergence_curve, '-rd', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerIndices', 1:10:maxIter, 'MarkerFaceColor', 'r');  

    % Figure aesthetics matching scientific publication standards
    xlabel('Iterations', 'FontSize', 16, 'FontWeight', 'bold');  
    ylabel('Average Fitness Value', 'FontSize', 16, 'FontWeight', 'bold');  
    title(['Average Convergence Curve of AOGWO on F', num2str(test_function_index)], 'FontSize', 16);  
    set(gca, 'FontSize', 14); 
    grid on;
    box on;
    hold off;  
end

disp('===========================================================');
disp('AOGWO Benchmark Evaluation Successfully Completed.');
disp('===========================================================');