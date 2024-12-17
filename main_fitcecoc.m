%----------------------------------------------------------------
% File:     main_fitcecoc.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Fri Nov 22 20:02:05 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% Classification into several classes
% This script trains a facial recognition model. The model
% is saved to a .MAT file, along with necessary data to perform facial
% recognition:

targetSize = [128,128];
k=40;                                   % Number of features to consider
location = fullfile('lfw');

disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

disp('Creating subset of several persons...');
persons = {'Angelina_Jolie', 'Eduardo_Duhalde', 'Amelie_Mauresmo', 'Abdullah_Gul',...
    'Akhmed_Zakayev', 'Cherie_Blair', 'Cesar_Maia', 'Wen_Ho_Lee', 'Mark_Kelly', 'Phil_McGraw', ...
    'Alvaro_Uribe', 'Ana_Palacio', 'Andre_Agassi', 'Andy_Roddick', 'Ann_Veneman',...
    'Anna_Kournikova', 'Ariel_Sharon', 'Arnold_Schwarzenegger', 'Atal_Bihari_Vajpayee',...
    'Ben_Affleck', 'Bill_Clinton', 'Bill_Gates', 'Bill_Simon', 'Britney_Spears', ...
    'Carlos_Menem', 'Carlos_Moya', 'Catherine_Zeta-Jones', 'Charles_Moose', ...
    'Colin_Powell', 'David_Nalbandian', 'Dick_Cheney', 'Dominique_de_Villepin',...
    'Donald_Rumsfeld', 'Edmund_Stoiber', 'Fidel_Castro', 'George_HW_Bush',...
    'George_Robertson', 'Gerhard_Schroeder', 'Gloria_Macapagal_Arroyo', ...
    'Marcelo_Bielsa', 'Debra_Brown', 'Theresa_May', 'Gene_Robinson', 'Ben_Wallace', 'Gretchen_Mol', ...
    'Halle_Berry', 'Hamid_Karzai', 'Hans_Blix', 'Hillary_Clinton', 'Hu_Jintao',...
    'Jim_Talent', 'Simon_Cowell', 'Peter_Mackay', 'AJ_Cook', 'Casey_Mears', 'Viola_Davis', ...
    'Jeb_Bush', 'Jennifer_Aniston', 'Jennifer_Capriati', 'Jennifer_Garner',...
    'John_Bolton', 'John_Howard', 'John_Kerry', 'John_Negroponte', 'John_Snow', ...
    'Joschka_Fischer', 'Jose_Maria_Aznar', 'Juan_Carlos_Ferrero', 'Julianne_Moore', ...
    'Luciano_Pavarotti', 'Sadie_Frost', 'Chuck_Yeager', 'Peter_Medgyessy', 'Lois_Smart', 'Tonga', ...
    'Lindsay_Davenport', 'Lleyton_Hewitt', 'Lucio_Gutierrez', 'Luiz_Inacio_Lula_da_Silva', ...
    'Johnny_Depp', 'Ratna_Sari_Dewi_Sukarno', 'Keith_Bogans', 'Bob_Beauprez', 'Billy_Rork', ...
    'Mahmoud_Abbas', 'Mark_Philippoussis', 'Megawati_Sukarnoputri', 'Meryl_Streep', ...
    'Michael_Bloomberg', 'Michael_Jackson', 'Michael_Schumacher', 'Mike_Weir', 'Mohammad_Khatami',...
    'Mohammed_Al-Douri', 'Muhammad_Ali', 'Nancy_Pelosi', 'Naomi_Watts', 'Nestor_Kirchner', ...
    'Nicanor_Duarte_Frutos', 'Nicole_Kidman', 'Norah_Jones', 'Paradorn_Srichaphan', ...
    'Paul_Bremer', 'Paul_Wolfowitz', 'Pervez_Musharraf', 'Pete_Sampras', 'Pierce_Brosnan',...
    'Queen_Elizabeth_II', 'Recep_Tayyip_Erdogan', 'Renee_Zellweger', 'Ricardo_Lagos', 'Richard_Gephardt', ...
    'Gilberto_Simoni', 'Mike_Matheny', 'John_Spencer', 'Prince_Harry', 'Jean_Carnahan', ...
    'Saddam_Hussein', 'Salma_Hayek', 'Serena_Williams', 'Sergey_Lavrov', 'Silvio_Berlusconi', ... 
    };


[lia, locb] = ismember(imds0.Labels, persons);
imds = subset(imds0, lia);

t=tiledlayout('flow');
nexttile(t);
montage(imds);

disp('Reading all images');
A = readall(imds);

B = cat(3,A{:});
D = prod(targetSize);
B = reshape(B,D,[]);

disp('Normalizing data...');
B = single(B)./256;
% NOTE: Normalization subtracts the mean pixel value
% from all pixels and divides by standard deviation. It is
% equivalent to:
%     [B,C,SD] = normalize(B, 1)
% This procedure is different from an alternative:
%     [B,C,SD] = normalize(B, 2)
% which computes the 'mean face' and subtracts it from every
% face. SD is then the l^2-norm between a face and mean face.
[B,C,SD] = normalize(B);
tic;
[U,S,V] = svd(B,'econ');
toc;

% Get an montage of eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);

nexttile(t);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
colormap(gray);

% NOTE: Rows of V are observations, columns are features.
% Observations need to be in rows.
k = min(size(V,2),k);

% Discard unnecessary data
W = S * V';                             % Transform V to weights (ala PCA)
W = W(1:k,:);                           % Keep first K weights
% NOTE: We will never again need singular values S
%S = diag(S);
%S = S(1:k);
U = U(:,1:k);                           % Keep K eigenfaces

% Find feature vectors of all images
X = W';
Y = categorical(imds.Labels, persons);

% Create colormap
cm=[1,0,0;
    0,0,1,
    0,1,0];
% Assign colors to target values
c=cm(1+mod(uint8(Y),size(cm,1)),:);

disp('Training Support Vector Machine...');
options = statset('UseParallel',true);
tic;

% You may try this, to get a more optimized model
% 'OptimizeHyperparameters','all',...

Mdl = fitcecoc(X, Y, 'Verbose', 2, 'Learners', 'svm', 'Options', options);
toc;

% Generate a plot in feature space using top two features
% First plot - top 3 predictors
figure; % Create a new figure window
scatter3(X(:, 1), X(:, 2), X(:, 3), 50, c);
title('A top 3-predictor plot');
xlabel('x1');
ylabel('x2');
zlabel('x3');

% Second plot - next 3 predictors
figure; % Create a new figure window
scatter3(X(:, 4), X(:, 5), X(:, 6), 50, c);
title('A next 3-predictor plot');
xlabel('x4');
ylabel('x5');
zlabel('x6');

% Get predictions and scores
[YPred, Score, Cost] = resubPredict(Mdl);

% ROC plot
disp('Plotting ROC metrics...');
rm = rocmetrics(imds.Labels, Score, persons);

figure; % Create a new figure window for ROC plot
plot(rm);

% Confusion matrix plot
disp('Plotting confusion matrix...');
figure; % Create a new figure window for the confusion matrix
confusionchart(Y, YPred);
title(['Number of features: ', num2str(k)]);

% Save the model and persons that the model recognizes.
% NOTE: An important part of the submission.
save('model','Mdl','persons','U', 'targetSize');
