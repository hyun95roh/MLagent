import streamlit as st
import time 
from proba_theory import gaussian

# Table of Content
st.sidebar.title('Contents:')

# 페이지 선택
page = st.sidebar.radio('Choose the page to read.',
                        ('Home', 'Probability Theory','Regression', 'Classification', 'Clustering','Dimension Reduction'))
if page == 'Home':
    # First progress bar to introduce Home page.
    progress_text = "You'll get to the page soon."
    my_bar = st.progress(0, text=progress_text) 
    for percent_complete in range(100): 
        time.sleep(0.01) 
        my_bar.progress(percent_complete +1, text=progress_text) 
    time.sleep(1)
    my_bar.empty()

if page == 'Probability Theory':
    st.title("Probability Theory")
    st.markdown("On this page, you can interact with various distributions.")

    # Generate Gaussian distribution object
    your_distribution = gaussian() 
    your_distribution.generate_data()
    fig = your_distribution.create_plot('Standard Normal Distribution') 
    plot_1 = st.plotly_chart(fig, use_container_width=True) 
    # Latex
    st.latex('\
    Z = \t{coef_{pi}/(pi) * exp(-(coef_{x} * x^{xpower} + coef_y * y^{ypower}) * coef_{exp} + bias_{exp}}\
    ')

    # Slider
    st.markdown("Turn on the toggle below(↓) and interact with your plot!")
    toggle_1 = st.toggle("Show configurations on sidebar")
    if toggle_1 : 
        with st.sidebar : 
            num_1 = st.number_input("Set min value of the slider",)
            num_2 = st.number_input("Set max value of the slider",1.0)
            your_distribution.coef_pi = st.slider('coef_pi', num_1, num_2, value=0.5, step=0.1 )
            your_distribution.coef_exp = st.slider('coef_exp',num_1, num_2, value=0.5, step=0.1 )
            your_distribution.coef_x = st.slider('coef_x',num_1, num_2, value=1.0, step=0.1 )
            your_distribution.coef_y = st.slider('coef_y',num_1, num_2, value=1.0, step=0.1 )
            your_distribution.exp_bias = st.slider('bias_exp',num_1, num_2, value=0.0, step=0.1 )
            if st.button('RESET',type="primary"): 
                your_distribution.coef_pi, your_distribution.coef_exp = 0.5, 0.5 
                your_distribution.coef_x, your_distribution.coef_y = 1.0, 1.0
                your_distribution.exp_bias = 0.0 
    
    your_distribution.generate_data()
    fig = your_distribution.create_plot('Modified Gaussian Plot') 
    plot_2 = st.plotly_chart(fig, use_container_width=True) 


elif page == 'Regression':
    st.title("Regression")
    st.markdown('# 1.Linear regression')
    with st.expander('Core interest', expanded=False): 
        '''
        1. Is at least one of the predictors(X_1 ~ X_p) useful in predicting the response?
        2. Do all the predictors help to explain Y, or is only a subset of the predictors useful?
        3. How well does the model fit the data? 
        4. Given a set of predictor values, what response value should we predict and how accurate is our prediction?
        5. What potential problems could undermine your model? 
        6. Can you discover any bias-variance trade-off?
        '''

    with st.expander('Problems and Remedies',expanded=False): 
        '''
        1. Non-linearity of response-predictor relationships.
            \n 
            -> Influence: Omitted variable bias(OVB) 
            \n
            -> Dignosis tool: Residual Plot (X-axis: y_hat / Y-axis: Residual)
            \n
            -> Remedy: (1)Bring non-linear model. (2)Find out omitted variables and add them to stabilize your model 
            \n
        2. Correlation of error terms (a.k.a. *Autocorrelation*)
            \n
            -> Dignosis tool : Residual Plot 
            \n
            -> Remedy: 
            \n
        3. Non-constant variance of error terms (a.k.a. *heteroskedasticity*)
            \n
            -> Influence: 
            \n
            -> Dignosis tool: Residual Plot for model fit 
            \n
            -> Remedy: Transform response Y using concave function(ex. sqrt(y), log(y), etc.)
            or fit your model by *weighted least squares*. 

        4. Outliers
            \n
            -> Influence: 
            \n 
            -> Dignosis tool: (Standard)Residual plot
            \n
            -> Remedy: Remove outliers. Existence of outliers may indeicate there are bunch of missing values.
            If missing data is the cause, try interpolation.

        5. High-leverage points 
            \n
            -> Influnece: 
            \n
            -> Dignosis tool: Leverage statistics(h_i)
            \n
            -> Remedy: Check out Residuals vs h_i scatter plot(Y-axis: Residual)
              and remove outliers. 

        6. Collinearity (Corrleation within features)
            \n
            -> Influence: Omitted Variable Bias(OVB)
            \n
            -> Dignosis tool: Scatter plot of features, VIF(Variance Inflation Factor), 
            Contour plots for the beta's RSS values
            \n
            -> Remedy: If multi-collinearity does not give any influence on the variance and unbiasedness of the key parameter, 
            you don't have to resolve it. 
        '''


elif page == 'Classification': 
    st.title("Coming soon")

elif page == 'Clustering': 
    st.title("Coming soon")

elif page == 'Dimension Reduction': 
    st.title("Coming soon")

