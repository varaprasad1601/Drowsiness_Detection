import login
import register
import call

print("1. Register\n2. Login")
key = input("Enter Choice :")
if key == "1":
    user = register.user_register()
elif key == "2":
    user = login.user_login()
    if user:
        details = user[0].split("-")
        action = call.call_to_user(details[1])
else:
    print("Enter Valid Choice")
