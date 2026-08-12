import json
import numpy as np
import pandas as pd
from scipy.special import logit
from hxmap.config import FEATURES, OUTPUT_DIR, FIGURE_DIR
from hxmap.train import artifacts_exist, fit_and_save_artifacts, load_artifacts
from hxmap.mapping import (
    get_feature_importance,
    get_meta_coefficients,
    get_reference_values,
    build_fma_mapping,
    build_interpolators,
    compute_total_cf,
    fit_total_cf_to_stacking_logit,
    map_total_cf_to_probability,
)
from hxmap.evaluate import youden_threshold, classification_metrics, calibration_intercept_slope, bootstrap_performance
from hxmap.plotting import plot_feature_points, plot_points_probability_curve


def main():
    if not artifacts_exist():
        print('Required artifacts missing. Training prerequisite models...')
        fit_and_save_artifacts()

    artifacts = load_artifacts()
    split = artifacts['split_data']
    X_train, X_test = split['X_train'], split['X_test']
    y_train, y_test = split['y_train'], split['y_test']
    id_test = split['id_test']

    base_models = {
        'rf': artifacts['rf_model'],
        'et': artifacts['et_model'],
        'catboost': artifacts['catboost_model'],
    }
    stacking_model = artifacts['stacking_model']

    feature_importances_df = get_feature_importance(base_models, X_train, y_train)
    meta_coeffs, meta_intercept = get_meta_coefficients(stacking_model)
    reference_values = get_reference_values(X_train)
    fma_mapping_df, minimum_cf_by_feature, largest_range = build_fma_mapping(base_models, feature_importances_df, meta_coeffs, X_train, reference_values)
    cf_functions, point_functions = build_interpolators(fma_mapping_df)

    train_total_cf = compute_total_cf(X_train, cf_functions)
    test_total_cf = compute_total_cf(X_test, cf_functions)

    train_stacking_prob = stacking_model.predict_proba(X_train)[:, 1]
    test_stacking_prob = stacking_model.predict_proba(X_test)[:, 1]
    lp_intercept, lp_slope = fit_total_cf_to_stacking_logit(train_total_cf, train_stacking_prob)

    train_lp, train_prob = map_total_cf_to_probability(train_total_cf, lp_intercept, lp_slope)
    test_lp, test_prob = map_total_cf_to_probability(test_total_cf, lp_intercept, lp_slope)
    test_total_points = np.asarray([sum(float(point_functions[f](row[f])) for f in FEATURES) for _, row in X_test.iterrows()])

    threshold = youden_threshold(y_train, train_prob)
    perf = classification_metrics(y_test, test_prob, threshold)
    cal_intercept, cal_slope = calibration_intercept_slope(y_test, test_prob)
    perf['Calibration intercept'] = cal_intercept
    perf['Calibration slope'] = cal_slope
    perf['Original stacking test AUC'] = classification_metrics(y_test, test_stacking_prob, 0.5)['AUC']

    bootstrap_df = bootstrap_performance(y_test, test_prob, threshold)
    perf_df = pd.DataFrame({'Metric': list(perf.keys()), 'Value': list(perf.values())})

    sum_minimum_cf = sum(minimum_cf_by_feature[f] for f in FEATURES)
    maximum_total_points = fma_mapping_df.groupby('Feature')['Nomogram Points'].max().sum()
    total_points_grid = np.linspace(0, maximum_total_points, 500)
    total_cf_grid = total_points_grid * largest_range / 100 + sum_minimum_cf
    total_lp_grid, total_probability_grid = map_total_cf_to_probability(total_cf_grid, lp_intercept, lp_slope)
    points_probability_df = pd.DataFrame({'Total Points': total_points_grid, 'Total Delta CF': total_cf_grid, 'Linear Predictor': total_lp_grid, 'Predicted Probability': total_probability_grid})

    threshold_total_cf = (logit(np.clip(threshold, 1e-8, 1 - 1e-8)) - lp_intercept) / lp_slope
    threshold_total_points = ((threshold_total_cf - sum_minimum_cf) / largest_range * 100) if largest_range > 0 else np.nan

    test_predictions_df = X_test.copy()
    test_predictions_df['Observed outcome'] = np.asarray(y_test)
    test_predictions_df['Total FMA contribution'] = test_total_cf
    test_predictions_df['Total nomogram points'] = test_total_points
    test_predictions_df['Predicted probability'] = test_prob
    test_predictions_df['Original stacking probability'] = test_stacking_prob
    test_predictions_df['Predicted class'] = (test_prob >= threshold).astype(int)
    test_predictions_df['ID'] = id_test.values

    reference_df = pd.DataFrame({'Feature': FEATURES, 'Reference value': [reference_values[f] for f in FEATURES]})
    mapping_parameters_df = pd.DataFrame({'Parameter': ['LP mapper intercept', 'LP mapper slope', 'Sum of feature minimum Delta CF', 'Largest contribution range', 'Maximum total points', 'Probability threshold', 'Corresponding points cutoff'], 'Value': [lp_intercept, lp_slope, sum_minimum_cf, largest_range, maximum_total_points, threshold, threshold_total_points]})

    fma_mapping_df.to_csv(OUTPUT_DIR / 'hxmap_feature_value_mapping.csv', index=False)
    feature_importances_df.to_csv(OUTPUT_DIR / 'hxmap_feature_importance.csv', index=False)
    reference_df.to_csv(OUTPUT_DIR / 'hxmap_reference_values.csv', index=False)
    test_predictions_df.to_csv(OUTPUT_DIR / 'hxmap_test_predictions.csv', index=False)
    perf_df.to_csv(OUTPUT_DIR / 'hxmap_test_performance.csv', index=False)
    bootstrap_df.to_csv(OUTPUT_DIR / 'hxmap_bootstrap_results.csv', index=False)
    points_probability_df.to_csv(OUTPUT_DIR / 'hxmap_points_probability_curve.csv', index=False)
    mapping_parameters_df.to_csv(OUTPUT_DIR / 'hxmap_mapping_parameters.csv', index=False)

    with pd.ExcelWriter(OUTPUT_DIR / 'HXMAP_complete_results.xlsx', engine='openpyxl') as writer:
        fma_mapping_df.to_excel(writer, sheet_name='Feature value mapping', index=False)
        feature_importances_df.to_excel(writer, sheet_name='Feature importance', index=False)
        reference_df.to_excel(writer, sheet_name='Reference values', index=False)
        test_predictions_df.to_excel(writer, sheet_name='Test predictions', index=False)
        perf_df.to_excel(writer, sheet_name='Test performance', index=False)
        bootstrap_df.to_excel(writer, sheet_name='Bootstrap results', index=False)
        points_probability_df.to_excel(writer, sheet_name='Points probability curve', index=False)
        mapping_parameters_df.to_excel(writer, sheet_name='Mapping parameters', index=False)

    plot_feature_points(fma_mapping_df, FIGURE_DIR)
    plot_points_probability_curve(points_probability_df, FIGURE_DIR / 'HXMAP_points_probability_curve.png')

    summary = {
        'status': 'ok',
        'strict_base_models': ['RandomForest', 'CatBoost', 'ExtraTrees'],
        'threshold': threshold,
        'threshold_total_points': threshold_total_points,
        'lp_mapper_intercept': lp_intercept,
        'lp_mapper_slope': lp_slope,
    }
    (OUTPUT_DIR / 'hxmap_run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('HX-MAP strict pipeline ready.')


if __name__ == '__main__':
    main()
