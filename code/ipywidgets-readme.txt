Interactive App: Regression Class Filtering

Youtube: https://youtu.be/f70RUYQiP0Y

Requires: rpy2 (which will require R and a bunch of R dependencies such as urbanmapr) [plus a slew of linux dependencies] to name a few.  I recommend running all python/R inside linux or WSL, but it is not a hard requirement.  I used windows env initially myself, but I have ported this over to wsl/linux.  To run as a web app, you need voila.

1st chart is a zca to non linear transformation 
	(see https://gist.github.com/thistleknot/13e8630ed9a50359c08b301c53f38ec6)


1st drop down is
	pipeline: scaled, ZCA whitened values (in addition to non linear transformed)

	ZCA != non ZCA rank order.

	ZCA = decorrelated data
		i.e. traverses columns and disentangtles correlations.

	'Plot each column w ZCA'
		shows this relationship 
			i.e. a non decorrelated column has with itself 
				once ZCA whitened has been applied

	[partial] correlation matrix of ZCA whitened data
		for curiosities sake

2nd drop down is ***model building***
	y = independent term
	x = deselect variables you don't wish to be included in model
		I tend to deselect population, and white
		But before I do that, 
			let's look at what the default model outputs for income

		Looks like it's going to include population?
		You can deselect autoremove and the model will simply select
			what you have chosen

Choose your y, deselect x variables you don't wish to include.

You can see the sorted [non linear] transformed response variable plotted here

Followed by a dataframe [of nonlinear transformed variables] 
	sorted by the response variable 
		with a heatmap applied.

Beneath the heatmap
	printout of variables removed from consideration 
		using partial correlation backstep filtering 
			(i.e. removing the least significant variable
				...
				reiterating)

	SHAP values

	Plot of response variable along with predicted response 
		using the regression equation at the end

	good models using this data
		Poverty
		Income
		University

	partial correlation matrix

	correlation matrix

	residual histogram

	ability to sort/plot [regression] residual, prediction vs actual

	regression diagnostics for the model (at the end)
	
	sns pairplot (scatterplot matrix)

AboveCenter/BelowCenter is used to bifurcate records by "class inclusion" 
	using the regression coefficients as filters 
		(i.e. + sign means > mean, - sign means < mean)

Center: [Checked: Mean, Unchecked: Median]
	Controls what center to use for bifurcation class inclusion method

Threshold is used for prediction and raw map methods
	used to control the cutoff score

3rd dropdown
	bifurcation = class inclusion/filtering using means and coefficient signs
	prediction = regression prediction given equation at end
	raw = raw response variables (in this case, 
		non linear transformed, but rank* order is preserved with original data)
			*I only ZCA whiten dependent terms
				i.e. doesn't affect rank order of Y

	Note boxplots of segmented data.
		Classes are:
			ingroup
			outgroup
			neither

	printout of group records sorted by independent term.

	Map

		Note: bifurcation color scheme doesn't match the others
			(green vs blue)
			but the 3rd group is always neither

Regression Equation
	Note: all significant terms
	Thanks to use of ZCA whitening, 
		variables are non biased (all have equal error estimates)
		variables are independent of each other [from correlation]
			except with Y
	This is because ZCA unit scales each variable.

Often resets the page as it recieves new requests to process due to form changes