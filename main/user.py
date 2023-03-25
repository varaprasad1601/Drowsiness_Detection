import login
import register

print("1. Register\n2. Login")
key = int(input("Enter Choice :"))
if key == 1:
    user = register.user_register()
elif key == 2:
    user = login.user_login()
else:
    print("Enter Valid Choice")
