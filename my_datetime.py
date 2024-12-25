import datetime
import pytz

# --- Working with Dates ---
tday = datetime.date.today()
print(f"Today's date: {tday}")

# Days until a specific date (e.g., Birthday)
bday = datetime.date(2024, 12, 22)
till_bday = bday - tday
print(f"Days until birthday: {till_bday.days}")

# --- Working with Timedeltas ---
tdelta = datetime.timedelta(hours=12)
print(f"12 hours from today: {tday + tdelta}")  # Note: Adding hours to a date object doesn't make sense

# --- Working with Time ---
t = datetime.time(9, 30, 45, 100000)
print(f"Time object: {t}")

# --- Working with Datetime ---
dt = datetime.datetime(2016, 7, 24, 12, 30, 45, tzinfo=pytz.UTC)
print(f"Datetime with UTC timezone: {dt}")

dt_utcnow = datetime.datetime.now(tz=pytz.UTC)
print(f"Current UTC time (using now): {dt_utcnow}")

dt_utcnow2 = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
print(f"Current UTC time (using utcnow): {dt_utcnow2}")

# Convert UTC to a different timezone (e.g., Mountain Time)
dt_mtn = dt_utcnow.astimezone(pytz.timezone('US/Mountain'))
print(f"UTC to Mountain Time: {dt_mtn}")

# Localize naive datetime to Mountain Time
dt_mtn = datetime.datetime.now()  # Current local time
mtn_tz = pytz.timezone('US/Mountain')
dt_mtn = mtn_tz.localize(dt_mtn)
print(f"Localized Mountain Time: {dt_mtn}")

# Convert Mountain Time to Eastern Time
dt_east = dt_mtn.astimezone(pytz.timezone('US/Eastern'))
print(f"Mountain to Eastern Time: {dt_east}")

# --- Formatting and Parsing Dates ---
# Format datetime to string
formatted_date = dt_mtn.strftime('%B %d, %Y')
print(f"Formatted date: {formatted_date}")

# Parse string to datetime
dt_str = 'July 24, 2016'
dt = datetime.datetime.strptime(dt_str, '%B %d, %Y')
print(f"Parsed datetime from string: {dt}")
