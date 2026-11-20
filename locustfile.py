from locust import HttpUser, task, between


class MiBiciUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict_p1(self):
        payload = {
            "departures_lag_1h": 2.0,
            "departures_lag_24h": 3.0,
            "departures_lag_168h": 4.0,
            "departures_rolling_mean_24h": 3.0,
            "hour_of_day": 14,
            "day_of_week": 0,
            "month": 8,
            "is_weekend": 0,
        }

        with self.client.post(
            "/predict",
            json=payload,
            name="P1 - Regression",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            else:
                try:
                    if "predicted_departures" not in response.json():
                        response.failure("Missing prediction")
                except ValueError:
                    response.failure("Invalid JSON response")

    @task(1)
    def predict_p3(self):
        payload = {
            "total_departures": 23,
            "total_arrivals": 10,
            "morning_departures": 20,
            "weekend_departures": 3,
            "dpcapacity": 0,
            "avg_daily_departures": 0,
            "morning_share": 0.5,
            "weekend_share": 0.2,
            "latitude": 0,
            "longitude": 0,
        }

        with self.client.post(
            "/predict/p3",
            json=payload,
            name="P3 - Clustering",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            else:
                try:
                    if "predicted_cluster" not in response.json():
                        response.failure("Missing cluster")
                except ValueError:
                    response.failure("Invalid JSON response")
