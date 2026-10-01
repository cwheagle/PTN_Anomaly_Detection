import pytest
import json
from unittest.mock import patch, MagicMock
from src.pipeline.window_state import WindowStateManager

@pytest.fixture
def mock_redis():
    with patch('src.pipeline.window_state.redis.Redis') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

def test_add_event(mock_redis):
    # given
    manager = WindowStateManager(host="dummy")
    key = "equip_A:port_1"
    event_data = {"in_packet": 100, "out_packet": 200}
    max_size = 3
    
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline
    
    # when
    manager.add_event(key, event_data, max_size=max_size)
    
    # then
    expected_json = json.dumps(event_data)
    mock_pipeline.lpush.assert_called_once_with(key, expected_json)
    mock_pipeline.ltrim.assert_called_once_with(key, 0, max_size - 1)
    mock_pipeline.execute.assert_called_once()

def test_get_window_order(mock_redis):
    # given
    manager = WindowStateManager(host="dummy")
    key = "equip_A:port_1"
    
    # Redis lrange returns newest first (index 0 is newest)
    raw_redis_data = [
        json.dumps({"time": "10:15"}), # Newest
        json.dumps({"time": "10:00"}),
        json.dumps({"time": "09:45"})  # Oldest
    ]
    mock_redis.lrange.return_value = raw_redis_data
    
    # when
    window = manager.get_window(key)
    
    # then
    mock_redis.lrange.assert_called_once_with(key, 0, -1)
    assert len(window) == 3
    # Should be reversed to chronological order (oldest first)
    assert window[0]["time"] == "09:45"
    assert window[1]["time"] == "10:00"
    assert window[2]["time"] == "10:15"
