psf_folder = 'D:\\JHU\\BDD\\SpineDetect-Clean\\Data\\Bootstrap\\NAOMi';

filePattern = fullfile(psf_folder, '*.mat'); 
fileList = dir(filePattern);

psf_idxs = 10:20:810;
psfs = zeros([length(psf_idxs) 100 100 250]);
masks = zeros([length(psf_idxs) 508 508]);

for k = 1:length(psf_idxs)
    idx = psf_idxs(k);
    fullFileName = fullfile(fileList(k).folder, sprintf("calc_psf_%d.mat", idx));

    load(fullFileName);

    psfs(k,:,:,:) = PSF_struct.psf;
    masks(k,:,:) = PSF_struct.mask;

end


%%

n_col = 10;
n_row = 2;

figure()

for yy = 1:n_row
    for xx = 1:n_col

        idx = ((yy-1) * n_col) + xx;
        display(idx)

        psf = squeeze(psfs(idx * 2,50,:,:))';

        subplot(n_row, n_col, idx)
        imagesc(psf)
        axis image

    end
end

%%

n_col = 5;
n_row = 4;

figure()

for yy = 1:n_row
    for xx = 1:n_col

        idx = ((yy-1) * n_col) + xx;
        mask = squeeze(masks(idx,:,:));

        subplot(n_row, n_col, idx)
        imagesc(mask)
        axis image

    end
end